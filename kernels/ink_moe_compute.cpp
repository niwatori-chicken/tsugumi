// Inkling decode MoE — compute (see ink_moe_reader.cpp).
//   gate_up, per expert k: g_k, u_k = x @ W_k (row 0) for the core's act column tile -> CB 4 / 5 (bf16)
//   act: on the dense tiles DG / DU (rows = experts, built by the writer, CB 8 / 9):
//        A = tanh(DG / gc) * sigmoid(DG) * tanh(DU / uc) -> CB 10 (the writer shares it with all cores)
//   down, per expert k: Y_k = D @ Wdn_k (row k of Y_k is expert k's output) for the 2 output tiles -> CB 11
//   combine: out = sum_k SEL_k @ Y_k (SEL_k: [0][k] = routing weight k) -> CB 16 (row 0)
// RT: 0 1/gc, 1 1/uc (float bits), 2 stage (debug: 3 = all), 3 half of K read first (the reader's order)
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

constexpr uint32_t c_X = 0, c_W = 1, c_D = 2, c_S = 3, c_G = 4, c_U = 5, c_DG = 8, c_DU = 9, c_A = 10, c_Y = 11,
                   c_O = 16;
constexpr uint32_t K = 8;

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

FORCE_INLINE void zero2() {
    fill_tile_init();
    fill_tile(0, 0.0f);
    fill_tile(1, 0.0f);
}

void kernel_main() {
    const uint32_t igc = get_arg_val<uint32_t>(0), iuc = get_arg_val<uint32_t>(1);
    // matmul: in0 -> SrcB, in1 -> SrcA, so the formats are configured (in1, in0) (bfp8 weights, bf16 activations)
    compute_kernel_hw_startup(c_W, c_X, c_G);
    const uint32_t stage = get_arg_val<uint32_t>(2), first = get_arg_val<uint32_t>(3);
    cb_wait_front(c_X, 72);
    for (uint32_t k = 0; k < K; ++k) {
        tile_regs_acquire();
        zero2();
        reconfig_data_format(c_W, c_X);
        matmul_init(c_X, c_W, 0);
        for (uint32_t g = 0; g < 6; ++g) {
            cb_wait_front(c_W, 24);
            if (stage != 4) {
                const uint32_t kt0 = ((g + 3 * first) % 6) * 12;  // halves in the reader's order
                for (uint32_t i = 0; i < 12; ++i) {
                    matmul_tiles(c_X, c_W, kt0 + i, 2 * i, 0);
                    matmul_tiles(c_X, c_W, kt0 + i, 2 * i + 1, 1);
                }
            }
            cb_pop_front(c_W, 24);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, c_G);
        pack_to(1, c_U);
        tile_regs_release();
    }
    cb_pop_front(c_X, 72);
    if (stage == 1) return;
    // act on the dense tiles
    cb_wait_front(c_DG, 1);
    cb_wait_front(c_DU, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(c_DG);
    copy_init(c_DG);
    copy_tile(c_DG, 0, 0);
    copy_tile(c_DG, 0, 1);
    reconfig_data_format_srca(c_DU);
    copy_init(c_DU);
    copy_tile(c_DU, 0, 2);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, igc);
    mul_unary_tile(2, iuc);
    tanh_tile_init();
    tanh_tile(0);
    tanh_tile(2);
    sigmoid_tile_init();
    sigmoid_tile(1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    mul_binary_tile(0, 2, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, c_A);
    tile_regs_release();
    cb_pop_front(c_DG, 1);
    cb_pop_front(c_DU, 1);
    if (stage == 2) return;
    // down
    cb_wait_front(c_D, 36);
    for (uint32_t k = 0; k < K; ++k) {
        tile_regs_acquire();
        zero2();
        reconfig_data_format(c_W, c_D);
        matmul_init(c_D, c_W, 0);
        for (uint32_t g = 0; g < 6; ++g) {
            cb_wait_front(c_W, 12);
            if (stage != 4) {
                const uint32_t kt0 = ((g + 3 * first) % 6) * 6;
                for (uint32_t i = 0; i < 6; ++i) {
                    matmul_tiles(c_D, c_W, kt0 + i, 2 * i, 0);
                    matmul_tiles(c_D, c_W, kt0 + i, 2 * i + 1, 1);
                }
            }
            cb_pop_front(c_W, 12);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, c_Y);
        pack_to(1, c_Y);
        tile_regs_release();
    }
    cb_pop_front(c_D, 36);
    // combine
    cb_wait_front(c_S, K);
    cb_wait_front(c_Y, 2 * K);
    tile_regs_acquire();
    zero2();
    reconfig_data_format(c_Y, c_S);
    matmul_init(c_S, c_Y, 0);
    for (uint32_t k = 0; k < K; ++k) {
        matmul_tiles(c_S, c_Y, k, 2 * k, 0);
        matmul_tiles(c_S, c_Y, k, 2 * k + 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, c_O);
    pack_to(1, c_O);
    tile_regs_release();
    cb_pop_front(c_S, K);
    cb_pop_front(c_Y, 2 * K);
}
