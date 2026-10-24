// single source of truth for the fusions supported by the Metal backend
//
// every fusable subgraph is declared exactly once as a lm_ggml_metal_fusion entry in
// the table in ggml-metal-fusion.cpp. both the graph optimizer (lm_ggml_metal_fusion_max)
// and the op encoders (lm_ggml_metal_fusion_next) consult this same table with the same device
// properties. a check that passes in FULL mode also passes in STRUCTURAL mode, so the encoders
// fuse only groups that the optimizer may pack.

#pragma once

#include "ggml-impl.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lm_ggml_metal_device_props;

// the maximum number of nodes that can be fused in a single kernel
// (also the maximum length of a packed fusion group during graph optimization)
#define LM_GGML_METAL_FUSION_MAX 16

typedef enum lm_ggml_metal_fusion_mode {
    // structural checks for the graph optimizer, which runs before the graph tensors are allocated (weights
    // already are); a check may skip conditions that differ between batch sizes, and so accept more than FULL
    LM_GGML_METAL_FUSION_STRUCTURAL = 0,
    // full checks, including buffer placement; used by the op encoders
    LM_GGML_METAL_FUSION_FULL,
} lm_ggml_metal_fusion_mode;

// identifier of each fusion pattern so the op encoders know which kernel to use
typedef enum lm_ggml_metal_fusion_id {
    LM_GGML_METAL_FUSION_NONE = 0,
    LM_GGML_METAL_FUSION_NORM_MUL,     // NORM/RMS_NORM + MUL
    LM_GGML_METAL_FUSION_NORM_MUL_ADD, // NORM/RMS_NORM + MUL + ADD
    LM_GGML_METAL_FUSION_NORM_SCALE,   // NORM/RMS_NORM + SCALE
    LM_GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N (N in [2, 7])
    LM_GGML_METAL_FUSION_SNAKE,        // MUL + SIN + SQR + MUL + ADD
    LM_GGML_METAL_FUSION_GDN_CACHE,    // GATED_DELTA_NET + CPY (write snapshots into the recurrent cache)
    LM_GGML_METAL_FUSION_TOPK_MOE,     // SOFT_MAX + ARGSORT + GET_ROWS + norm/scale (MoE routing)
    LM_GGML_METAL_FUSION_MOE_REDUCE,   // MUL + expert VIEWs + ADD chain (MoE output reduction)
    LM_GGML_METAL_FUSION_SSM_CONV_SILU, // SSM_CONV + UNARY (silu)
    LM_GGML_METAL_FUSION_MUL_MAT_ADD,  // MUL_MAT + ADD (residual added in the few-row MMA store)
} lm_ggml_metal_fusion_id;

struct lm_ggml_metal_fusion; // defined in ggml-metal-fusion.cpp

typedef struct lm_ggml_metal_fusion lm_ggml_metal_fusion;

// access the fusion identifier without exposing the full pattern definition
lm_ggml_metal_fusion_id lm_ggml_metal_fusion_get_id(const struct lm_ggml_metal_fusion * fusion);

// apply any alloc-dependencies required by the fused kernels during graph optimize
void lm_ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, struct lm_ggml_tensor *, struct lm_ggml_tensor *),
        struct lm_ggml_cgraph * gf);

// ---- shared fusion info ---------------------------------------------------

// shared fusion debugging context, owned by the device; newly created backend contexts for that
// device register with it so the fusion counters are race-free and accumulate across contexts.
struct lm_ggml_metal_fusion_info; // defined in ggml-metal-fusion.cpp

struct lm_ggml_metal_fusion_info * lm_ggml_metal_fusion_info_init(bool enabled, int debug);
void lm_ggml_metal_fusion_info_free(struct lm_ggml_metal_fusion_info * finfo);

bool lm_ggml_metal_fusion_info_enabled(const struct lm_ggml_metal_fusion_info * finfo);
bool lm_ggml_metal_fusion_info_stats  (const struct lm_ggml_metal_fusion_info * finfo);
int  lm_ggml_metal_fusion_info_debug  (const struct lm_ggml_metal_fusion_info * finfo);

int          lm_ggml_metal_fusion_info_n_fusions(const struct lm_ggml_metal_fusion_info * finfo);
const char * lm_ggml_metal_fusion_info_label    (const struct lm_ggml_metal_fusion_info * finfo, int idx);
uint64_t     lm_ggml_metal_fusion_info_count    (const struct lm_ggml_metal_fusion_info * finfo, int idx);

void lm_ggml_metal_fusion_info_count_fusion(struct lm_ggml_metal_fusion_info * finfo, const struct lm_ggml_metal_fusion * fusion);
void lm_ggml_metal_fusion_info_set_enabled (struct lm_ggml_metal_fusion_info * finfo, bool enabled);

void lm_ggml_metal_fusion_info_stats_init (      struct lm_ggml_metal_fusion_info * finfo);
void lm_ggml_metal_fusion_info_stats_reset(      struct lm_ggml_metal_fusion_info * finfo);
int  lm_ggml_metal_fusion_info_stats_get  (const struct lm_ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n);
void lm_ggml_metal_fusion_info_labels_init(      struct lm_ggml_metal_fusion_info * finfo);

// compute phase: longest fusion starting at idx (a position in node_idxs) that matches in `mode` on a device with
// props. returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes consumed.
const lm_ggml_metal_fusion * lm_ggml_metal_fusion_next(
        const struct lm_ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        lm_ggml_metal_fusion_mode mode,
        int * n_out);

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused on a device with props, chaining patterns back-to-back. returns at least 1.
int lm_ggml_metal_fusion_max(const struct lm_ggml_cgraph * gf, int idx);

#ifdef __cplusplus
}
#endif
