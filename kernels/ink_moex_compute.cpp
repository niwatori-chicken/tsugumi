// Inkling prefill MoE for long chunks — compute (see ink_moex_reader.cpp). Per entry (expert, <= 32 CTM tokens):
//   tilize the tokens' rows of X (CB 0, 2 ct blocks of 32 x 1152) -> X (CB 4, tile (m, kt) at m * 72 + kt);
//   gate_up on the ct compact row tiles -> CG / CU (CB 8 / 9);
//   act = tanh(G / gc) * sigmoid(G) * tanh(U / uc) * routing weight (CB 14) -> SHARE (CB 13, CTM slots);
// after the batch barrier, per entry: down on its compact act (CB 2) -> Y (CB 18, 2 CTM tiles), then the scatter:
//   for each row tile m the entry touches, ACC[m] += PS(m, :) @ Y with the writer's placement tiles (CB 3) by
//   packer L1 accumulation into ACC (CB 15, 2 MT float32 tiles, zeroed first); at the end ACC -> OUT (CB 16, bf16).
// CB 12 CTL: word 0 = U, 1 + 2j = n_j, 2 + 2j = mask of row tiles touched by entry j.
// RT: 0 1/gc, 1 1/uc (float bits), 2 half of K read first, 3 NB, 4 MT, 5 dbg (2: no matmuls)
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_STG = 0, c_W = 1, c_D = 2, c_PS = 3, c_X = 4, c_CG = 8, c_CU = 9, c_A = 10, c_CTL = 12,
                   c_SH = 13, c_WC = 14, c_ACC = 15, c_O = 16, c_Y = 18;

#ifdef TRISC_UNPACK
#define TSC() (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#define PROF(i, ...)                \
    {                               \
        const uint32_t _t = TSC();  \
        __VA_ARGS__;                \
        prof[i] += TSC() - _t;      \
    }
#else
#define TSC() 0u
#define PROF(i, ...) __VA_ARGS__;
#endif
uint32_t prof[16];

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
    const uint32_t CTM = MT < 2 ? MT : 2;
    compute_kernel_hw_startup(c_W, c_X, c_CG);
    const uint32_t t_start = TSC();
    for (uint32_t i = 0; i < 16; ++i) prof[i] = 0;
    // ACC = 0
    cb_reserve_back(c_ACC, 2 * MT);
    pack_reconfig_data_format(c_ACC);
    for (uint32_t i = 0; i < 2 * MT; ++i) {
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        done();
        pack_tile<true>(0, c_ACC, i);
        tile_regs_release();
    }
    cb_wait_front(c_CTL, 1);
    const uint32_t U = read_tile_value(c_CTL, 0, 0);
    const uint32_t nbatch = (U + NB - 1) / NB;
    // software pipeline: gate_up / act of batch b + 1 before the down of batch b (hides the barrier and the
    // act-tile gather)
    auto gate_up_batch = [&](uint32_t b) {
        const uint32_t j0 = b * NB, n = (j0 + NB < U) ? NB : U - j0;
        for (uint32_t jj = 0; jj < n; ++jj) {
            const uint32_t ct = (read_tile_value(c_CTL, 0, 1 + 2 * (j0 + jj)) + 31) / 32;
            // the tokens' rows of X -> compact tiles
            // (blocks of 36 tiles in order: tile (m, kt) at m * 72 + kt; fixed 72 CTM tiles per entry)
            reconfig_data_format_srca(c_STG);
            pack_reconfig_data_format(c_X);
            tilize_init(c_STG, 36, c_X);
            for (uint32_t blk = 0; blk < 2 * CTM; ++blk) {
                cb_reserve_back(c_X, 36);
                if (blk < 2 * ct) {
                    PROF(2, cb_wait_front(c_STG, 36));
                    if (!(dbg & 16)) tilize_block(c_STG, 36, c_X);
                    cb_pop_front(c_STG, 36);
                }
                cb_push_back(c_X, 36);
            }
            tilize_uninit(c_STG, c_X);
            cb_wait_front(c_X, 72 * CTM);
            // gate_up: dst 2m / 2m + 1 = gate / up of compact row tile m
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * ct; ++i) fill_tile(i, 0.0f);
            reconfig_data_format(c_W, c_X);
            matmul_block_init(c_X, c_W, 0, 2, ct, 72);
            for (uint32_t g = 0; g < 6; ++g) {
                PROF(1, cb_wait_front(c_W, 24));
                const uint32_t kt0 = ((g + 3 * first) % 6) * 12;
                if (!(dbg & 2))
                    for (uint32_t i = 0; i < 12; ++i) matmul_block(c_X, c_W, kt0 + i, 2 * i, 0, 0, 2, ct, 72);
                cb_pop_front(c_W, 24);
            }
            done();
            cb_reserve_back(c_CG, CTM);  // fixed CTM-tile blocks (ct used)
            cb_reserve_back(c_CU, CTM);
            pack_reconfig_data_format(c_CG);
            for (uint32_t m = 0; m < ct; ++m) {
                pack_tile<true>(2 * m, c_CG, m);
                pack_tile<true>(2 * m + 1, c_CU, m);
            }
            cb_push_back(c_CG, CTM);
            cb_push_back(c_CU, CTM);
            tile_regs_release();
            cb_pop_front(c_X, 72 * CTM);
            // act on the compact tiles
            cb_wait_front(c_CG, CTM);
            cb_wait_front(c_CU, CTM);
            PROF(6, cb_wait_front(c_WC, CTM));
            for (uint32_t m = 0; m < ct && !(dbg & 64); ++m) {
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
                if (!(dbg & 256)) {
                    tanh_tile_init();
                    tanh_tile(0);
                    tanh_tile(2);
                }
                if (!(dbg & 512)) {
                    sigmoid_tile_init();
                    sigmoid_tile(1);
                }
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
            for (uint32_t m = (dbg & 64) ? 0 : ct; m < CTM; ++m) {  // the entry's remaining share slots (never read)
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
    };
    auto down_batch = [&](uint32_t b) {
        const uint32_t j0 = b * NB, n = (j0 + NB < U) ? NB : U - j0;
        for (uint32_t jj = 0; jj < n; ++jj) {
            const uint32_t ct = (read_tile_value(c_CTL, 0, 1 + 2 * (j0 + jj)) + 31) / 32;
            const uint32_t mask = read_tile_value(c_CTL, 0, 2 + 2 * (j0 + jj));
            // down -> Y (compact tile (mi, nn) at 2 mi + nn)
            PROF(3, cb_wait_front(c_D, 36 * CTM));
            tile_regs_acquire();
            fill_tile_init();
            for (uint32_t i = 0; i < 2 * ct; ++i) fill_tile(i, 0.0f);
            reconfig_data_format(c_W, c_D);
            matmul_block_init(c_D, c_W, 0, 2, ct, 36);
            for (uint32_t g = 0; g < 6; ++g) {
                PROF(4, cb_wait_front(c_W, 12));
                const uint32_t kt0 = ((g + 3 * first) % 6) * 6;
                if (!(dbg & 2))
                    for (uint32_t i = 0; i < 6; ++i) matmul_block(c_D, c_W, kt0 + i, 2 * i, 0, 0, 2, ct, 36);
                cb_pop_front(c_W, 12);
            }
            done();
            cb_reserve_back(c_Y, 2 * CTM);
            pack_reconfig_data_format(c_Y);
            for (uint32_t i = 0; i < 2 * ct; ++i) pack_tile<true>(i, c_Y, i);
            cb_push_back(c_Y, 2 * CTM);
            tile_regs_release();
            cb_pop_front(c_D, 36 * CTM);
            // scatter-add into ACC
            cb_wait_front(c_Y, 2 * CTM);
            reconfig_data_format(c_Y, c_PS);
            matmul_init(c_PS, c_Y, 0);
            for (uint32_t m = 0; m < MT; ++m) {
                if (!((mask >> m) & 1)) continue;
                PROF(5, cb_wait_front(c_PS, CTM));
                if (!(dbg & 32)) {
                    tile_regs_acquire();
                    for (uint32_t nn = 0; nn < 2; ++nn)
                        for (uint32_t mi = 0; mi < ct; ++mi) matmul_tiles(c_PS, c_Y, mi, 2 * mi + nn, nn);
                    done();
                    pack_reconfig_data_format(c_ACC);
                    pack_reconfig_l1_acc(1);
                    pack_tile<true>(0, c_ACC, 2 * m);
                    pack_tile<true>(1, c_ACC, 2 * m + 1);
                    pack_reconfig_l1_acc(0);
                    tile_regs_release();
                }
                cb_pop_front(c_PS, CTM);
            }
            cb_pop_front(c_Y, 2 * CTM);
        }
    };
    if (nbatch) gate_up_batch(0);
    for (uint32_t b = 0; b < nbatch; ++b) {
        if (b + 1 < nbatch) gate_up_batch(b + 1);
        down_batch(b);
    }
#ifdef TRISC_UNPACK
    {  // (profiling) counters -> CB 21 for the writer
        prof[0] = TSC() - t_start;
        volatile uint32_t* pl = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(21).fifo_rd_ptr << 4);
        for (uint32_t i = 0; i < 16; ++i) pl[i] = prof[i];
    }
#endif
    cb_push_back(c_ACC, 2 * MT);
    cb_wait_front(c_ACC, 2 * MT);
    reconfig_data_format_srca(c_ACC);
    copy_init(c_ACC);
    for (uint32_t i = 0; i < 2 * MT; ++i) {
        tile_regs_acquire();
        copy_tile(c_ACC, i, 0);
        done();
        pack_to(0, c_O);
        tile_regs_release();
    }
    cb_pop_front(c_ACC, 2 * MT);
    cb_pop_front(c_CTL, 1);
}
