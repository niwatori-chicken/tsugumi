// Inkling prefill MoE with compact token tiles — compute (see ink_moec_reader.cpp). Per batch of NB used experts:
//   gate_up per expert on all MT row tiles: G_m, U_m -> CB 4 / 5 (the writer packs the routed tokens' rows into
//     CT compact tiles CG / CU, CB 8 / 9);
//   act on the compact tiles: A = tanh(G / gc) * sigmoid(G) * tanh(U / uc) * routing weights (CB 14, column
//     broadcast) -> CB 13 SHARE (MT slots per expert: CT act tiles, then zero tiles);
//   down per expert on its compact act (CB 2, 36 x CT tiles): Y = A @ Wdn (CT x 2 tiles) -> CB 18;
//   scatter: out[m] += sum_m' PS(m, m') @ Y(m') with the writer's 0/1 placement tiles PS (CB 3, MT x CT per
//     expert: [t % 32][r % 32] = 1 for token t = L[r]); the running sum in CB 15 (float32) between batches; the
//     last batch -> CB 16 (bf16, 2 MT output tiles).
// RT: 0 1/gc, 1 1/uc (float bits), 2 half of K read first, 3 NB, 4 MT (2 MT float32 DST tiles: MT = 4 needs
//     the full-sync DST), 5 dbg (2: no matmuls). Entries (expert, <= 32 CTM tokens), CTM = min(MT, 2).
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

constexpr uint32_t c_X = 0, c_W = 1, c_D = 2, c_PS = 3, c_G = 4, c_U = 5, c_CG = 8, c_CU = 9, c_A = 10, c_CTL = 12,
                   c_SH = 13, c_WC = 14, c_ACC = 15, c_O = 16, c_Y = 18;

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
    const uint32_t dbg = get_arg_val<uint32_t>(5);
    const uint32_t CTM = MT < 2 ? MT : 2;  // compact tiles per entry (fixed CB blocks)
    compute_kernel_hw_startup(c_W, c_X, c_G);
    cb_wait_front(c_CTL, 1);
    const uint32_t U = read_tile_value(c_CTL, 0, 0);
    cb_wait_front(c_X, 72 * MT);
    const uint32_t nbatch = (U + NB - 1) / NB;
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t j0 = b * NB, n = (j0 + NB < U) ? NB : U - j0;
        for (uint32_t jj = 0; jj < n; ++jj) {
            const uint32_t ct = (read_tile_value(c_CTL, 0, 1 + j0 + jj) + 31) / 32;
            tile_regs_acquire();  // dst 2m / 2m + 1: gate / up of row tile m
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * MT; ++i) fill_tile(i, 0.0f);
            reconfig_data_format(c_W, c_X);
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
            // act on the compact tiles (rows = the routed tokens)
            cb_wait_front(c_CG, CTM);  // fixed CTM-tile blocks (variable ones could wrap around a CB's end): ct used
            cb_wait_front(c_CU, CTM);
            cb_wait_front(c_WC, CTM);
            for (uint32_t m = 0; m < ct; ++m) {
                tile_regs_acquire();
                reconfig_data_format_srca(c_CG);
                copy_init(c_CG);
                copy_tile(c_CG, m, 0);
                copy_tile(c_CG, m, 1);
                reconfig_data_format_srca(c_CU);
                copy_init(c_CU);
                copy_tile(c_CU, m, 2);
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
                done();
                pack_to(0, c_A);
                tile_regs_release();
                cb_wait_front(c_A, 1);
                tile_regs_acquire();
                reconfig_data_format(c_A, c_WC);
                mul_bcast_cols_init(c_A, c_WC);
                mul_tiles_bcast_cols(c_A, c_WC, 0, m, 0);
                done();
                pack_to(0, c_SH);
                tile_regs_release();
                cb_pop_front(c_A, 1);
            }
            for (uint32_t m = ct; m < CTM; ++m) {  // the entry's remaining share slots (never read)
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, 0.0f);
                done();
                pack_to(0, c_SH);
                tile_regs_release();
            }
            cb_pop_front(c_CG, CTM);
            cb_pop_front(c_CU, CTM);
            cb_pop_front(c_WC, CTM);
        }
        // down per expert on its compact act -> Y (CB 18)
        for (uint32_t jj = 0; jj < n; ++jj) {
            const uint32_t ct = (read_tile_value(c_CTL, 0, 1 + j0 + jj) + 31) / 32;
            cb_wait_front(c_D, 36 * CTM);
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * ct; ++i) fill_tile(i, 0.0f);
            reconfig_data_format(c_W, c_D);
            matmul_block_init(c_D, c_W, 0, 2, ct, 36);
            for (uint32_t g = 0; g < 6; ++g) {
                cb_wait_front(c_W, 12);
                const uint32_t kt0 = ((g + 3 * first) % 6) * 6;
                if (!(dbg & 2))
                    for (uint32_t i = 0; i < 6; ++i) matmul_block(c_D, c_W, kt0 + i, 2 * i, 0, 0, 2, ct, 36);
                cb_pop_front(c_W, 12);
            }
            for (uint32_t i = 2 * ct; i < 2 * CTM; ++i) {  // block padding (never read)
                fill_tile_init();
                fill_tile(i, 0.0f);
            }
            done();
            for (uint32_t i = 0; i < 2 * CTM; ++i) pack_to(i, c_Y);
            tile_regs_release();
            cb_pop_front(c_D, 36 * CTM);
        }
        // scatter-add the batch's Y rows to the token rows
        tile_regs_acquire();  // dst 2m + nn: output tile nn of row tile m
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
        reconfig_data_format(c_Y, c_PS);
        matmul_init(c_PS, c_Y, 0);
        for (uint32_t jj = 0; jj < n; ++jj) {
            const uint32_t ct = (read_tile_value(c_CTL, 0, 1 + j0 + jj) + 31) / 32;
            cb_wait_front(c_PS, MT * CTM);
            cb_wait_front(c_Y, 2 * CTM);
            for (uint32_t m = 0; m < MT; ++m)
                for (uint32_t mi = 0; mi < ct; ++mi)
                    for (uint32_t nn = 0; nn < 2; ++nn)
                        matmul_tiles(c_PS, c_Y, m * CTM + mi, 2 * mi + nn, 2 * m + nn);
            cb_pop_front(c_PS, MT * CTM);
            cb_pop_front(c_Y, 2 * CTM);
        }
        done();
        const uint32_t out = (b + 1 == nbatch) ? c_O : c_ACC;
        for (uint32_t i = 0; i < 2 * MT; ++i) pack_to(i, out);
        tile_regs_release();
    }
    cb_pop_front(c_X, 72 * MT);
    cb_pop_front(c_CTL, 1);
}
