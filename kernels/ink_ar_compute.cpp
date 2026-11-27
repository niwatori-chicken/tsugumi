// Inkling decode: block attention residual + rms norm — compute (both phases).
//   T = QT + SH[nb] X [+ NB]        (rows 0..nb-1 blocks, row nb the raw h, rows 31.. the queries)
// phase 1: G_part = sum over the core's tiles of T T^T (float32) -> CB 8
// phase 2: G = sum of the NC1 partials -> CB 8 (the writer turns it into the weight tiles, CB 9:
//          W_k (row 0 = attention weights of query k), W2_k (W_k * rms factor of the output), FD (diag: 1 / rms(h)
//          at nb, 1 for the blocks)); then per tile: AI_k = W_k T -> CB 16 (without norm weights), XN_k = (W2_k T) * LN_k -> CB 17,
//          TOUT = FD T -> CB 18 (the block matrix with row nb normalized)
// RT: 0 phase, 1 NT, 2 use_nb, 3 NC1, 4 NQ, 5 use_ln, 6 tout
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_X = 0, c_N = 1, c_Q = 2, c_S = 3, c_P = 4, c_T = 5, c_L = 6, c_G = 8, c_W = 9, c_TL = 10,
                   c_AI = 16, c_XN = 17, c_TO = 18;

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

FORCE_INLINE void copy_to_dst(uint32_t cb, uint32_t idx, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, idx, dst);
}

FORCE_INLINE void done() {
    tile_regs_commit();
    tile_regs_wait();
}

FORCE_INLINE void mm_to(uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t out) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(a, b);
    matmul_init(a, b, 0);
    matmul_tiles(a, b, ai, bi, 0);
    done();
    pack_to(0, out);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t phase = get_arg_val<uint32_t>(0), NT = get_arg_val<uint32_t>(1), use_nb = get_arg_val<uint32_t>(2);
    const uint32_t NC1 = get_arg_val<uint32_t>(3), NQ = get_arg_val<uint32_t>(4), use_ln = get_arg_val<uint32_t>(5);
    const uint32_t tout = get_arg_val<uint32_t>(6);
    compute_kernel_hw_startup(c_S, c_X, c_T);
    if (phase == 2) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < NC1; ++i) {
            cb_wait_front(c_P, 1);
            copy_to_dst(c_P, 0, i == 0 ? 0 : 1);
            if (i > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(c_P, 1);
        }
        done();
        pack_to(0, c_G);
        tile_regs_release();
    }
    cb_wait_front(c_S, 1);
    cb_wait_front(c_X, NT);
    cb_wait_front(c_Q, NT);
    if (use_nb) cb_wait_front(c_N, NT);
    for (uint32_t j = 0; j < NT; ++j) {
        tile_regs_acquire();
        copy_to_dst(c_Q, j, 0);
        reconfig_data_format(c_S, c_X);
        matmul_init(c_S, c_X, 0);
        matmul_tiles(c_S, c_X, 0, j, 0);
        if (use_nb) {
            copy_to_dst(c_N, j, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        done();
        pack_to(0, c_T);
        tile_regs_release();
    }
    cb_pop_front(c_S, 1);
    cb_pop_front(c_X, NT);
    cb_pop_front(c_Q, NT);
    if (use_nb) cb_pop_front(c_N, NT);
    cb_wait_front(c_T, NT);
    if (phase == 1) {
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        reconfig_data_format(c_T, c_T);
        matmul_init(c_T, c_T, 1);
        for (uint32_t j = 0; j < NT; ++j) matmul_tiles(c_T, c_T, j, j, 0);
        done();
        pack_to(0, c_G);
        tile_regs_release();
        cb_pop_front(c_T, NT);
        return;
    }
    const uint32_t nw = 2 * NQ + 1;
    cb_wait_front(c_W, nw);
    for (uint32_t j = 0; j < NT; ++j) {
        for (uint32_t k = 0; k < NQ; ++k) {
            if (!use_ln) mm_to(c_W, k, c_T, j, c_AI);
            if (use_ln) {  // (W2_k T) * LN_k
                mm_to(c_W, NQ + k, c_T, j, c_TL);
                cb_wait_front(c_TL, 1);
                cb_wait_front(c_L, 1);
                tile_regs_acquire();
                reconfig_data_format(c_TL, c_L);
                mul_init(c_TL, c_L, false);
                mul_tiles(c_TL, c_L, 0, 0, 0);
                done();
                pack_to(0, c_XN);
                tile_regs_release();
                cb_pop_front(c_TL, 1);
                cb_pop_front(c_L, 1);
            }
        }
        if (tout) mm_to(c_W, 2 * NQ, c_T, j, c_TO);
    }
    cb_pop_front(c_W, nw);
    cb_pop_front(c_T, NT);
}
