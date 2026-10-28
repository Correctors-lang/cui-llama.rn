#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<lm_ggml_op> lm_ggml_metal_fusion_filter_ops(const std::vector<lm_ggml_op> & ops_all) {
    std::vector<lm_ggml_op> ops;

    for (lm_ggml_op op : ops_all) {
        if (!lm_ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct lm_ggml_metal_fusion {
    lm_ggml_metal_fusion_id id;

    std::vector<lm_ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<lm_ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + lm_ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of lm_ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct lm_ggml_metal_fusion   * fusion,
                  const struct lm_ggml_tensor * const * nodes,
                  const struct lm_ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        lm_ggml_metal_fusion_mode       mode);

    lm_ggml_metal_fusion(
            lm_ggml_metal_fusion_id id,
            const std::vector<lm_ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct lm_ggml_metal_fusion   * fusion,
                          const struct lm_ggml_tensor * const * nodes,
                          const struct lm_ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                lm_ggml_metal_fusion_mode       mode))
        : id(id),
          ops(lm_ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

lm_ggml_metal_fusion_id lm_ggml_metal_fusion_get_id(const lm_ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool lm_ggml_metal_fusion_same_buffer(const lm_ggml_tensor * a, const lm_ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    lm_ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    lm_ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    lm_ggml_metal_buffer_t ca = (lm_ggml_metal_buffer_t) ba->context;
    lm_ggml_metal_buffer_t cb = (lm_ggml_metal_buffer_t) bb->context;

    return lm_ggml_metal_buffer_get_id(ca, a).metal == lm_ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool lm_ggml_metal_fusion_overlap(const lm_ggml_tensor * a, const lm_ggml_tensor * b) {
    lm_ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    lm_ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const lm_ggml_metal_buffer_id bid_a = lm_ggml_metal_buffer_get_id((lm_ggml_metal_buffer_t) ba->context, a);
    const lm_ggml_metal_buffer_id bid_b = lm_ggml_metal_buffer_get_id((lm_ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < lm_ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < lm_ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool lm_ggml_metal_fusion_check_norm(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(mode);
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);

    LM_GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == LM_GGML_METAL_FUSION_NORM_SCALE) {
        LM_GGML_ASSERT(fusion->ops.size() == 2);

        const lm_ggml_tensor * scale = nodes[1];
        if (scale->op != LM_GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != LM_GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!lm_ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != LM_GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool lm_ggml_metal_fusion_check_ssm_conv_silu(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(fusion);
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);
    LM_GGML_UNUSED(mode);

    const lm_ggml_tensor * conv = nodes[0];
    const lm_ggml_tensor * un   = nodes[1];

    if (conv->op != LM_GGML_OP_SSM_CONV || un->op != LM_GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (lm_ggml_get_unary_op(un) != LM_GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != LM_GGML_TYPE_F32 || un->type != LM_GGML_TYPE_F32 || !lm_ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool lm_ggml_metal_fusion_check_add_chain(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);
    LM_GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!lm_ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == LM_GGML_METAL_FUSION_FULL) {
            if (!lm_ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors lm_ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool lm_ggml_metal_fusion_check_gdn_cache(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(fusion);
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);

    const lm_ggml_tensor * gdn = nodes[0];
    const lm_ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != LM_GGML_TYPE_F32 || (gdn->flags & LM_GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != LM_GGML_OP_CPY || (cpy->flags & LM_GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = lm_ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = lm_ggml_row_size(LM_GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const lm_ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const lm_ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != LM_GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !lm_ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[LM_GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != LM_GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + LM_GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != lm_ggml_type_size(LM_GGML_TYPE_F32) ||
        dst->nb[1] != lm_ggml_row_size(LM_GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == LM_GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool lm_ggml_metal_fusion_check_snake(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(fusion);
    LM_GGML_UNUSED(mode);
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);

    const lm_ggml_tensor * mul0     = nodes[0];
    const lm_ggml_tensor * sin_node = nodes[1];
    const lm_ggml_tensor * sqr      = nodes[2];
    const lm_ggml_tensor * mul1     = nodes[3];
    const lm_ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const lm_ggml_tensor * x = lm_ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const lm_ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const lm_ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const lm_ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == LM_GGML_TYPE_F32 || x->type == LM_GGML_TYPE_F16 || x->type == LM_GGML_TYPE_BF16) &&
        (a->type    == LM_GGML_TYPE_F32) && (inv_b->type    == LM_GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = lm_ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        lm_ggml_is_contiguous(x) && lm_ggml_is_contiguous(add) &&
        lm_ggml_is_contiguous(a) && lm_ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define LM_GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
static const std::vector<lm_ggml_op> ops_topk_moe = {
    LM_GGML_OP_SOFT_MAX, LM_GGML_OP_RESHAPE, LM_GGML_OP_ARGSORT, LM_GGML_OP_VIEW, LM_GGML_OP_GET_ROWS
};
static const std::vector<lm_ggml_op> ops_topk_moe_scale = {
    LM_GGML_OP_SOFT_MAX, LM_GGML_OP_RESHAPE, LM_GGML_OP_ARGSORT, LM_GGML_OP_VIEW, LM_GGML_OP_GET_ROWS, LM_GGML_OP_SCALE
};
static const std::vector<lm_ggml_op> ops_topk_moe_norm = {
    LM_GGML_OP_SOFT_MAX, LM_GGML_OP_RESHAPE, LM_GGML_OP_ARGSORT, LM_GGML_OP_VIEW, LM_GGML_OP_GET_ROWS,
    LM_GGML_OP_RESHAPE, LM_GGML_OP_SUM_ROWS, LM_GGML_OP_CLAMP, LM_GGML_OP_DIV, LM_GGML_OP_RESHAPE
};
static const std::vector<lm_ggml_op> ops_topk_moe_norm_scale = {
    LM_GGML_OP_SOFT_MAX, LM_GGML_OP_RESHAPE, LM_GGML_OP_ARGSORT, LM_GGML_OP_VIEW, LM_GGML_OP_GET_ROWS,
    LM_GGML_OP_RESHAPE, LM_GGML_OP_SUM_ROWS, LM_GGML_OP_CLAMP, LM_GGML_OP_DIV, LM_GGML_OP_RESHAPE, LM_GGML_OP_SCALE
};

static bool lm_ggml_metal_fusion_check_topk_moe(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_ASSERT(fusion->ops.size() >= 3);
    LM_GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    const bool with_norm  = n_ops >= 6;
    const bool with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<lm_ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const lm_ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != LM_GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[LM_GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const lm_ggml_tensor * softmax        = gf->nodes[raw_start];
    const lm_ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    const lm_ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    const lm_ggml_tensor * ids            = gf->nodes[raw_start + 3];
    const lm_ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    const lm_ggml_tensor * out            = gf->nodes[raw_end];
    const lm_ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    float scale   = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    ((const int32_t *) softmax->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) softmax->op_params) + 1, sizeof(max_bias));
    if (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        return false;
    }

    if (logits->type != LM_GGML_TYPE_F32 || softmax->type != LM_GGML_TYPE_F32 ||
        out->type != LM_GGML_TYPE_F32 || ids->type != LM_GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > LM_GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > LM_GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!lm_ggml_is_contiguous(logits) || !lm_ggml_is_contiguous(out) ||
        ids->nb[0] != lm_ggml_type_size(LM_GGML_TYPE_I32) ||
        ids->nb[1] != lm_ggml_type_size(LM_GGML_TYPE_I32) * n_expert) {
        return false;
    }

    if (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const lm_ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        const lm_ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        const lm_ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        const lm_ggml_tensor * div              = gf->nodes[raw_start + 8];
        const lm_ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            return false;
        }

        if (with_scale) {
            const lm_ggml_tensor * scale_node = gf->nodes[raw_start + 10];
            if (scale_node->src[0] != out_reshaped) {
                return false;
            }
        }
    } else if (with_scale) {
        const lm_ggml_tensor * scale_node = gf->nodes[raw_start + 5];
        if (scale_node->src[0] != get_rows) {
            return false;
        }
    }

    const int outputs[2] = { raw_start + 3, raw_end };
    if (!lm_ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == LM_GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define LM_GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct lm_ggml_metal_moe_reduce_match {
    const lm_ggml_tensor * experts;
    const lm_ggml_tensor * weights;
    const lm_ggml_tensor * dst;
    int node_count;
};

static bool lm_ggml_metal_fusion_match_moe_reduce(
        const lm_ggml_cgraph * gf, int node_idx, const std::vector<lm_ggml_op> & ops_all,
        lm_ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const lm_ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != LM_GGML_OP_MUL || mul->type != LM_GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > LM_GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == LM_GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != LM_GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[LM_GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const lm_ggml_tensor * experts = mul->src[0];
    const lm_ggml_tensor * weights = mul->src[1];
    const lm_ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != LM_GGML_TYPE_F32 || weights->type != LM_GGML_TYPE_F32 || dst->type != LM_GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!lm_ggml_is_contiguous(experts) || !lm_ggml_is_contiguous(weights) || !lm_ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const lm_ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const lm_ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const lm_ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const lm_ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const lm_ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!lm_ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool lm_ggml_metal_fusion_check_moe_reduce(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(nodes);

    lm_ggml_metal_moe_reduce_match match;
    if (!lm_ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == LM_GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (lm_ggml_metal_fusion_overlap(match.dst, match.experts) || lm_ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool lm_ggml_metal_tensor_is_weight(const struct lm_ggml_tensor * t) {
    const lm_ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && lm_ggml_backend_buffer_get_usage(base->buffer) == LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct lm_ggml_tensor * lm_ggml_metal_mul_mat_add_operand(const struct lm_ggml_tensor * mm, const struct lm_ggml_tensor * add) {
    if (add->op != LM_GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const lm_ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == LM_GGML_TYPE_F32 && add->type == LM_GGML_TYPE_F32 && !lm_ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct lm_ggml_tensor * lm_ggml_metal_mul_mat_add_residual(const struct lm_ggml_tensor * mm, const struct lm_ggml_tensor * add) {
    const lm_ggml_tensor * res = lm_ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && lm_ggml_are_same_shape(res, mm) &&
        lm_ggml_is_contiguous(res) && lm_ggml_is_contiguous(mm) && lm_ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool lm_ggml_metal_fusion_check_mul_mat_add(
        const lm_ggml_metal_fusion       * fusion,
        const lm_ggml_tensor * const     * nodes,
        const lm_ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              lm_ggml_metal_fusion_mode    mode) {
    LM_GGML_UNUSED(gf);
    LM_GGML_UNUSED(node_idxs);
    LM_GGML_UNUSED(idx);
    LM_GGML_UNUSED(fusion);

    const lm_ggml_tensor * mm  = nodes[0];
    const lm_ggml_tensor * add = nodes[1];

    if (lm_ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !lm_ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == LM_GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const lm_ggml_tensor * res = lm_ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !lm_ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !lm_ggml_metal_fusion_overlap(add, mm->src[0]) && !lm_ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !lm_ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<lm_ggml_op> ops_norm_mul         = { LM_GGML_OP_NORM, LM_GGML_OP_MUL };
static const std::vector<lm_ggml_op> ops_norm_mul_add     = { LM_GGML_OP_NORM, LM_GGML_OP_MUL, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_norm_scale       = { LM_GGML_OP_NORM, LM_GGML_OP_SCALE };
static const std::vector<lm_ggml_op> ops_rms_norm_mul     = { LM_GGML_OP_RMS_NORM, LM_GGML_OP_MUL };
static const std::vector<lm_ggml_op> ops_rms_norm_mul_add = { LM_GGML_OP_RMS_NORM, LM_GGML_OP_MUL, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_rms_norm_scale   = { LM_GGML_OP_RMS_NORM, LM_GGML_OP_SCALE };

static const std::vector<lm_ggml_op> ops_add_2 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_add_3 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_add_4 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_add_5 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_add_6 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_add_7 = { LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD };
static const std::vector<lm_ggml_op> ops_snake = { LM_GGML_OP_MUL, LM_GGML_OP_SIN, LM_GGML_OP_SQR, LM_GGML_OP_MUL, LM_GGML_OP_ADD };

static const std::vector<lm_ggml_op> ops_gdn_cache = { LM_GGML_OP_GATED_DELTA_NET, LM_GGML_OP_CPY };

static const std::vector<lm_ggml_op> ops_ssm_conv_silu = { LM_GGML_OP_SSM_CONV, LM_GGML_OP_UNARY };

static const std::vector<lm_ggml_op> ops_moe_reduce_2 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_3 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_4 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW,
    LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_5 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW,
    LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_6 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW,
    LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_7 = {
    LM_GGML_OP_MUL, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW,
    LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};
static const std::vector<lm_ggml_op> ops_moe_reduce_8 = {
    LM_GGML_OP_MUL,
    LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW, LM_GGML_OP_VIEW,
    LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD, LM_GGML_OP_ADD
};

static const std::vector<lm_ggml_op> ops_mul_mat_add = { LM_GGML_OP_MUL_MAT, LM_GGML_OP_ADD };

static const std::vector<lm_ggml_metal_fusion> lm_ggml_metal_fusions = {
    { LM_GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     false, lm_ggml_metal_fusion_check_norm },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     false, lm_ggml_metal_fusion_check_add_chain },
    { LM_GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     false, lm_ggml_metal_fusion_check_snake },
    { LM_GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     true,  lm_ggml_metal_fusion_check_gdn_cache },
    { LM_GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    true,  lm_ggml_metal_fusion_check_topk_moe },
    { LM_GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    true,  lm_ggml_metal_fusion_check_topk_moe },
    { LM_GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    true,  lm_ggml_metal_fusion_check_topk_moe },
    { LM_GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    true,  lm_ggml_metal_fusion_check_topk_moe },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     true,  lm_ggml_metal_fusion_check_moe_reduce },
    { LM_GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     false, lm_ggml_metal_fusion_check_ssm_conv_silu },
    { LM_GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     false, lm_ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

static bool lm_ggml_metal_fusion_match_raw_pattern(
        const lm_ggml_cgraph * gf, int node_idx, const std::vector<lm_ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void lm_ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, lm_ggml_tensor *, lm_ggml_tensor *),
        lm_ggml_cgraph * gf,
        const lm_ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<lm_ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        lm_ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < LM_GGML_MAX_SRC; ++s) {
            lm_ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void lm_ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, lm_ggml_tensor *, lm_ggml_tensor *),
        lm_ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const lm_ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const lm_ggml_metal_fusion & fusion : lm_ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (lm_ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            lm_ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string lm_ggml_metal_fusion_label(const lm_ggml_metal_fusion * fusion) {
    LM_GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += lm_ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct lm_ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct lm_ggml_metal_fusion_info * lm_ggml_metal_fusion_info_init(bool enabled, int debug) {
    lm_ggml_metal_fusion_info * finfo = new lm_ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        lm_ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void lm_ggml_metal_fusion_info_free(lm_ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool lm_ggml_metal_fusion_info_enabled(const lm_ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool lm_ggml_metal_fusion_info_stats(const lm_ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int lm_ggml_metal_fusion_info_debug(const lm_ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int lm_ggml_metal_fusion_info_n_fusions(const lm_ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * lm_ggml_metal_fusion_info_label(const lm_ggml_metal_fusion_info * finfo, int idx) {
    LM_GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t lm_ggml_metal_fusion_info_count(const lm_ggml_metal_fusion_info * finfo, int idx) {
    LM_GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void lm_ggml_metal_fusion_info_count_fusion(lm_ggml_metal_fusion_info * finfo, const lm_ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - lm_ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void lm_ggml_metal_fusion_info_set_enabled(lm_ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void lm_ggml_metal_fusion_info_labels_init(lm_ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(lm_ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(lm_ggml_metal_fusions.size());

    for (const lm_ggml_metal_fusion & fusion : lm_ggml_metal_fusions) {
        finfo->labels.emplace_back(lm_ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void lm_ggml_metal_fusion_info_stats_init(lm_ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    lm_ggml_metal_fusion_info_labels_init(finfo);
}

void lm_ggml_metal_fusion_info_stats_reset(lm_ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int lm_ggml_metal_fusion_info_stats_get(const lm_ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool lm_ggml_metal_fusion_check_memory_ranges(
        const lm_ggml_metal_fusion      * fusion,
        const lm_ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const lm_ggml_tensor * a, const lm_ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const lm_ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const lm_ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const lm_ggml_tensor * a, const lm_ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + lm_ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + lm_ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const lm_ggml_tensor * src, const lm_ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const lm_ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const lm_ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < LM_GGML_MAX_SRC; ++s) {
                const lm_ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == LM_GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        LM_GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const lm_ggml_metal_fusion * lm_ggml_metal_fusion_next(
        const lm_ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        lm_ggml_metal_fusion_mode mode,
        int * n_out) {
    const lm_ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const lm_ggml_metal_fusion & fusion : lm_ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const lm_ggml_tensor * nodes[LM_GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!lm_ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[LM_GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                LM_GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!lm_ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == LM_GGML_METAL_FUSION_FULL &&
                !lm_ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int lm_ggml_metal_fusion_max(const lm_ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (lm_ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see lm_ggml_metal_fusion_filter_ops)
    int idxs[LM_GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < LM_GGML_METAL_FUSION_MAX; i++) {
        if (!lm_ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < LM_GGML_METAL_FUSION_MAX) {
        int len = 1;
        const lm_ggml_metal_fusion * fusion = lm_ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, LM_GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > LM_GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(LM_GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
