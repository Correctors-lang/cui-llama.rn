#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in misc.metal. Widths up to
// 512 run on the simdgroup kernel and need no threadgroup memory. The wider ones allocate
// float[N] per threadgroup, so they are only available where that fits.
static bool lm_ggml_metal_fwht_supported_size(int64_t n, size_t max_tg_mem) {
    if (n == 64 || n == 128 || n == 256 || n == 512) {
        return true;
    }

    if (n == 1024 || n == 2048 || n == 4096 || n == 8192) {
        return (size_t) n * sizeof(float) <= max_tg_mem;
    }

    return false;
}

// the FWHT kernels handle a Hadamard-hinted MUL_MAT only under these conditions. supports_op
// and the dispatch must ask the same question: an F16 src1 that is admitted but then falls
// through reaches the generic path, which has no F32 src0 by F16 src1 kernel.
bool lm_ggml_metal_op_mul_mat_use_fwht(const struct lm_ggml_tensor * op, size_t max_tg_mem) {
    return lm_ggml_get_op_params_i32(op, 1) == LM_GGML_HINT_SRC0_IS_HADAMARD && op->type == LM_GGML_TYPE_F32 &&
           (op->src[1]->type == LM_GGML_TYPE_F32 || op->src[1]->type == LM_GGML_TYPE_F16) && lm_ggml_is_contiguous(op->src[1]) &&
           lm_ggml_is_contiguous(op) && lm_ggml_are_same_shape(op->src[1], op) &&
           lm_ggml_metal_fwht_supported_size(op->src[1]->ne[0], max_tg_mem);
}

bool lm_ggml_metal_op_mul_mat_use_mm(const struct lm_ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !lm_ggml_is_transposed(op->src[0]) &&
           !lm_ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool lm_ggml_metal_op_mul_mat_id_use_mm(const struct lm_ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne02 <= 1024 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
static constexpr int64_t LM_GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
static constexpr int64_t LM_GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
static constexpr int64_t LM_GGML_METAL_MMA_K_CHUNK = 64;

enum lm_ggml_metal_mma_kind lm_ggml_metal_mul_mv_mma_kind(enum lm_ggml_type type, int rt) {
    if (type == LM_GGML_TYPE_Q4_0 || (type == LM_GGML_TYPE_Q8_0 && rt == 1)) {
        return LM_GGML_METAL_MMA_KIND_BLK;
    }
    return type == LM_GGML_TYPE_Q5_K ? LM_GGML_METAL_MMA_KIND_Q5_K : LM_GGML_METAL_MMA_KIND_GEN;
}

int lm_ggml_metal_mul_mv_mma_rt(const struct lm_ggml_tensor * op) {
    return op->src[1]->ne[1] > LM_GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

static bool lm_ggml_metal_mul_mv_mma_type_supported(enum lm_ggml_type type) {
    switch (type) {
        case LM_GGML_TYPE_F32:
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_BF16:
        case LM_GGML_TYPE_Q1_0:
        case LM_GGML_TYPE_Q2_0:
        case LM_GGML_TYPE_MXFP4:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

int64_t lm_ggml_metal_mul_mv_mma_k_step(enum lm_ggml_type type, int rt) {
    if (!lm_ggml_metal_mul_mv_mma_type_supported(type)) {
        return 0;
    }
    return lm_ggml_metal_mul_mv_mma_kind(type, rt) == LM_GGML_METAL_MMA_KIND_BLK ? lm_ggml_blck_size(type) : LM_GGML_METAL_MMA_K_CHUNK;
}

static bool lm_ggml_metal_mul_mat_mma_type_ok(const struct lm_ggml_tensor * op) {
    const lm_ggml_tensor * src0 = op->src[0];
    const int64_t step = lm_ggml_metal_mul_mv_mma_k_step(src0->type, lm_ggml_metal_mul_mv_mma_rt(op));

    return step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == lm_ggml_type_size(src0->type);
}

// the fewest src1 rows at which the few-row MMA kernels beat the mat-vec kernels (measured on an M3 Ultra)
static int64_t lm_ggml_metal_mul_mv_mma_rows_min(enum lm_ggml_type type) {
    switch (type) {
        case LM_GGML_TYPE_F32:
            return 6;
        case LM_GGML_TYPE_TQ2_0:
            return 5;
        case LM_GGML_TYPE_BF16:
            return 4;
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_Q2_0:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_MXFP4:
            return 3;
        default:
            return 2;
    }
}

bool lm_ggml_metal_op_mul_mat_use_mma(const struct lm_ggml_tensor * op) {
    const lm_ggml_tensor * src0 = op->src[0];
    const lm_ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    const bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    return lm_ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == LM_GGML_TYPE_F32 && src1->ne[1] >= lm_ggml_metal_mul_mv_mma_rows_min(src0->type) && src1->ne[1] <= LM_GGML_METAL_MMA_ROWS_MAX &&
        !lm_ggml_is_transposed(src0) && !lm_ggml_is_transposed(src1) &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

bool lm_ggml_metal_op_mul_mat_may_use_mma(const struct lm_ggml_tensor * op) {
    return lm_ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == LM_GGML_TYPE_F32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct lm_ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    lm_ggml_mem_range_type pt;
};

struct lm_ggml_mem_ranges {
    std::vector<lm_ggml_mem_range> ranges;

    int debug = 0;
};

lm_ggml_mem_ranges_t lm_ggml_mem_ranges_init(int debug) {
    auto * res = new lm_ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void lm_ggml_mem_ranges_free(lm_ggml_mem_ranges_t mrs) {
    delete mrs;
}

void lm_ggml_mem_ranges_reset(lm_ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool lm_ggml_mem_ranges_add(lm_ggml_mem_ranges_t mrs, lm_ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor(const lm_ggml_tensor * tensor, lm_ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    LM_GGML_ASSERT(!tensor->view_src);

    lm_ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with lm_ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + lm_ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor_src(const lm_ggml_tensor * tensor) {
    return lm_ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static lm_ggml_mem_range lm_ggml_mem_range_from_tensor_dst(const lm_ggml_tensor * tensor) {
    return lm_ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool lm_ggml_mem_ranges_add_src(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mr = lm_ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        LM_GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return lm_ggml_mem_ranges_add(mrs, mr);
}

static bool lm_ggml_mem_ranges_add_dst(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mr = lm_ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        LM_GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return lm_ggml_mem_ranges_add(mrs, mr);
}

bool lm_ggml_mem_ranges_add(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            lm_ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return lm_ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool lm_ggml_mem_ranges_check(lm_ggml_mem_ranges_t mrs, lm_ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                LM_GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool lm_ggml_mem_ranges_check_src(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mr = lm_ggml_mem_range_from_tensor_src(tensor);

    const bool res = lm_ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool lm_ggml_mem_ranges_check_dst(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);

    lm_ggml_mem_range mr = lm_ggml_mem_range_from_tensor_dst(tensor);

    const bool res = lm_ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool lm_ggml_mem_ranges_check(lm_ggml_mem_ranges_t mrs, const lm_ggml_tensor * tensor) {
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!lm_ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return lm_ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    lm_ggml_tensor * node;

    std::vector<lm_ggml_tensor *> fused;

    lm_ggml_op op() const {
        return node->op;
    }

    const lm_ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return lm_ggml_op_is_empty(node->op);
    }

    void add_fused(lm_ggml_tensor * t) {
        fused.push_back(t);
    }

    bool is_output(const lm_ggml_tensor * t) const {
        if (t == node) {
            return true;
        }
        for (const auto * f : fused) {
            if (t == f) {
                return true;
            }
        }
        return false;
    }
};

static std::vector<int> lm_ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](lm_ggml_mem_ranges_t mrs, const node_info & node) {
        // only external sources matter: sources produced by the fused group are internal
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            const lm_ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!lm_ggml_mem_ranges_add_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
                const lm_ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!lm_ggml_mem_ranges_add_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        // all fused tensors are produced by the fused kernel
        if (!lm_ggml_mem_ranges_add_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!lm_ggml_mem_ranges_add_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](lm_ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            const lm_ggml_tensor * src = node.node->src[i];
            if (src && !node.is_output(src)) {
                if (!lm_ggml_mem_ranges_check_src(mrs, src)) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
                const lm_ggml_tensor * src = fused->src[i];
                if (src && !node.is_output(src)) {
                    if (!lm_ggml_mem_ranges_check_src(mrs, src)) {
                        return false;
                    }
                }
            }
        }

        if (!lm_ggml_mem_ranges_check_dst(mrs, node.node)) {
            return false;
        }
        for (const auto * fused : node.fused) {
            if (!lm_ggml_mem_ranges_check_dst(mrs, fused)) {
                return false;
            }
        }

        return true;
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](lm_ggml_op op) {
        switch (op) {
            case LM_GGML_OP_MUL_MAT:
            case LM_GGML_OP_MUL_MAT_ID:
            case LM_GGML_OP_ROPE:
            case LM_GGML_OP_NORM:
            case LM_GGML_OP_RMS_NORM:
            case LM_GGML_OP_GROUP_NORM:
            case LM_GGML_OP_L2_NORM:
            case LM_GGML_OP_SUM_ROWS:
            case LM_GGML_OP_SSM_CONV:
            case LM_GGML_OP_SSM_SCAN:
            case LM_GGML_OP_CLAMP:
            case LM_GGML_OP_TRI:
            case LM_GGML_OP_DIAG:
            case LM_GGML_OP_MUL:
            case LM_GGML_OP_ADD:
            case LM_GGML_OP_SUB:
            case LM_GGML_OP_DIV:
            case LM_GGML_OP_GLU:
            case LM_GGML_OP_SCALE:
            case LM_GGML_OP_UNARY:
            case LM_GGML_OP_GET_ROWS:
            case LM_GGML_OP_SET_ROWS:
            case LM_GGML_OP_SET:
            case LM_GGML_OP_CPY:
            case LM_GGML_OP_CONT:
            case LM_GGML_OP_REPEAT:
                return true;
            default:
                return lm_ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    lm_ggml_mem_ranges_t mrs0 = lm_ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    lm_ggml_mem_ranges_t mrs1 = lm_ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            lm_ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            lm_ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    lm_ggml_mem_ranges_free(mrs0);
    lm_ggml_mem_ranges_free(mrs1);

    return res;
}

void lm_ggml_graph_optimize(lm_ggml_cgraph * gf) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        const int f = lm_ggml_metal_fusion_max(gf, i);

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 1; k < f; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = lm_ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
