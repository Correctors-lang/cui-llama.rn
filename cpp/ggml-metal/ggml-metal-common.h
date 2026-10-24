// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lm_ggml_tensor;
struct lm_ggml_cgraph;

enum lm_ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct lm_ggml_mem_ranges * lm_ggml_mem_ranges_t;

lm_ggml_mem_ranges_t lm_ggml_mem_ranges_init(int debug);
void lm_ggml_mem_ranges_free(lm_ggml_mem_ranges_t mrs);

// remove all ranges from the set
void lm_ggml_mem_ranges_reset(lm_ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool lm_ggml_mem_ranges_add(lm_ggml_mem_ranges_t mrs, const struct lm_ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool lm_ggml_mem_ranges_check(lm_ggml_mem_ranges_t mrs, const struct lm_ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void lm_ggml_graph_optimize(struct lm_ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and lm_ggml_metal_op_mul_mat*
bool lm_ggml_metal_op_mul_mat_use_fwht (const struct lm_ggml_tensor * op, size_t max_tg_mem);
bool lm_ggml_metal_op_mul_mat_use_mm   (const struct lm_ggml_tensor * op, bool has_simdgroup_mm);
bool lm_ggml_metal_op_mul_mat_id_use_mm(const struct lm_ggml_tensor * op, bool has_simdgroup_mm);

bool lm_ggml_metal_op_mul_mat_use_mma    (const struct lm_ggml_tensor * op);
bool lm_ggml_metal_op_mul_mat_may_use_mma(const struct lm_ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
enum lm_ggml_metal_mma_kind { LM_GGML_METAL_MMA_KIND_BLK, LM_GGML_METAL_MMA_KIND_Q5_K, LM_GGML_METAL_MMA_KIND_GEN };
enum lm_ggml_metal_mma_kind lm_ggml_metal_mul_mv_mma_kind(enum lm_ggml_type type, int rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
int lm_ggml_metal_mul_mv_mma_rt(const struct lm_ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t lm_ggml_metal_mul_mv_mma_k_step(enum lm_ggml_type type, int rt);

#ifdef __cplusplus
}
#endif
