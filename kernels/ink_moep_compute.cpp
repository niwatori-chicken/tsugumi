// Inkling prefill MoE (MT x 32 token rows) — compute (see ink_moep_reader.cpp). Per batch of NB used experts:
//   gate_up per expert: G, U = X @ W (32 rows) -> CB 4 / 5; A = tanh(G / gc) * sigmoid(G) * tanh(U / uc) -> CB 10,
//   A * routing weights (column broadcast, CB 14) -> CB 13 (SHARE: read by every core after the barrier);
//   down: acc += A_e @ Wdn_e over the batch's experts (A from CB 2, 36 tiles per expert), the running sum kept in
//   CB 15 (float32) between batches; the last batch -> CB 16 (bf16, 2 output tiles).
// RT: 0 1/gc, 1 1/uc (float bits), 2 half of K read first, 3 NB, 4 MT (row tiles; 2 MT float32 DST tiles)
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
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_X = 0, c_W = 1, c_D = 2, c_G = 4, c_U = 5, c_A = 10, c_CTL = 12, c_SH = 13, c_WC = 14,
                   c_ACC = 15, c_O = 16;

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

FORCE_INLINE void done() {
    tile_regs_commit();
    tile_regs_wait();
}

void kernel_main() {
    const uint32_t igc = get_arg_val<uint32_t>(0), iuc = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2), NB = get_arg_val<uint32_t>(3), MT = get_arg_val<uint32_t>(4);
    const uint32_t dbg = get_arg_val<uint32_t>(5);  // debug bits: 2 no matmuls
    compute_kernel_hw_startup(c_W, c_X, c_G);
    cb_wait_front(c_CTL, 1);
    const uint32_t U = read_tile_value(c_CTL, 0, 0);
    cb_pop_front(c_CTL, 1);
    cb_wait_front(c_X, 72 * MT);
    const uint32_t nbatch = (U + NB - 1) / NB;
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t n = (b * NB + NB < U) ? NB : U - b * NB;
        for (uint32_t j = 0; j < n; ++j) {
            tile_regs_acquire();  // dst 2m / 2m + 1: gate / up of row tile m
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * MT; ++i) fill_tile(i, 0.0f);
            reconfig_data_format(c_W, c_X);
            // one call per K tile: the MT x 2 block (row tiles x [gate, up]); X rows at stride 72, DST (m, n) = 2 m + n
            matmul_block_init(c_X, c_W, 0, 2, MT, 72);
            for (uint32_t g = 0; g < 6; ++g) {
                cb_wait_front(c_W, 24);
                const uint32_t kt0 = ((g + 3 * first) % 6) * 12;
                if (!(dbg & 2))
                    for (uint32_t i = 0; i < 12; ++i) matmul_block(c_X, c_W, kt0 + i, 2 * i, 0, 0, 2, MT, 72);
                cb_pop_front(c_W, 24);
            }
            done();
            for (uint32_t m = 0; m < MT; ++m) {
                pack_to(2 * m, c_G);
                pack_to(2 * m + 1, c_U);
            }
            tile_regs_release();
            cb_wait_front(c_G, MT);
            cb_wait_front(c_U, MT);
            cb_wait_front(c_WC, MT);
            for (uint32_t m = 0; m < MT; ++m) {
            tile_regs_acquire();
            reconfig_data_format_srca(c_G);
            copy_init(c_G);
            copy_tile(c_G, m, 0);
            copy_tile(c_G, m, 1);
            reconfig_data_format_srca(c_U);
            copy_init(c_U);
            copy_tile(c_U, m, 2);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, igc);
            mul_unary_tile(2, iuc);
            tanh_tile_init();  // (the fast approximate forms: -14% kernel time, 9% output error: not used)
            tanh_tile(0);
            tanh_tile(2);
            sigmoid_tile_init();
            sigmoid_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(0, 2, 0);
            done();
            pack_to(0, c_A);
            tile_regs_release();
            // A * routing weights of the tokens (rows)
            cb_wait_front(c_A, 1);
            cb_wait_front(c_WC, 1);
            tile_regs_acquire();
            reconfig_data_format(c_A, c_WC);
            mul_bcast_cols_init(c_A, c_WC);
            mul_tiles_bcast_cols(c_A, c_WC, 0, m, 0);
            done();
            pack_to(0, c_SH);
            tile_regs_release();
            cb_pop_front(c_A, 1);
            }
            cb_pop_front(c_G, MT);
            cb_pop_front(c_U, MT);
            cb_pop_front(c_WC, MT);
        }
        // down for the batch
        tile_regs_acquire();  // dst 2m / 2m + 1: output tiles 2c / 2c + 1 of row tile m
        if (b == 0) {
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * MT; ++i) fill_tile(i, 0.0f);
        } else {
            cb_wait_front(c_ACC, 2 * MT);
            reconfig_data_format_srca(c_ACC);
            copy_init(c_ACC);
            for (uint32_t i = 0; i < 2 * MT; ++i) copy_tile(c_ACC, i, i);
            cb_pop_front(c_ACC, 2 * MT);
        }
        for (uint32_t j = 0; j < n; ++j) {
            cb_wait_front(c_D, 36 * MT);
            reconfig_data_format(c_W, c_D);
            matmul_block_init(c_D, c_W, 0, 2, MT, 36);  // D rows at stride 36
            for (uint32_t g = 0; g < 6; ++g) {
                cb_wait_front(c_W, 12);
                const uint32_t kt0 = ((g + 3 * first) % 6) * 6;
                if (!(dbg & 2))
                    for (uint32_t i = 0; i < 6; ++i) matmul_block(c_D, c_W, kt0 + i, 2 * i, 0, 0, 2, MT, 36);
                cb_pop_front(c_W, 12);
            }
            cb_pop_front(c_D, 36 * MT);
        }
        done();
        const uint32_t out = (b + 1 == nbatch) ? c_O : c_ACC;
        for (uint32_t i = 0; i < 2 * MT; ++i) pack_to(i, out);
        tile_regs_release();
    }
    cb_pop_front(c_X, 72 * MT);
}
