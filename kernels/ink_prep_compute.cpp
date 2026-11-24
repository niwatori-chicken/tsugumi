// Inkling decode: per-head preparation of q / r / k / v — compute (see ink_prep_reader.cpp).
//   conv:  Y = sum_lag X_lag * (TAPS_lag [+ DYN_lag / 3 broadcast over the columns])   (residual / normalizer in TAPS)
//   scale: Y = X * SCALE                      (else Y = X)
//   ss = rowsum(Y^2)
//   clip:  f = thr * rsqrt(max(ss, thr^2)) [* tau]      (= min(thr / |Y|, 1) [* tau])
//   rms:   f = rsqrt(ss * inv_hw + eps)
//   out = Y * f (broadcast over the columns) -> CB 16
// RT: 0 nt, 1 conv, 2 dyn, 3 scale, 4 tau, 5 mode (0 clip, 1 rms), 6 thr, 7 thr^2, 8 inv_hw, 9 eps, 10 1/3 (all floats as bits)
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_X = 0, c_H = 1, c_T = 2, c_D = 3, c_S = 4, c_TAU = 5, c_SC = 6, c_D3 = 7, c_Y = 8, c_SQ = 9,
                   c_SS = 10, c_F = 11, c_F2 = 12, c_O = 16;

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

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0), conv = get_arg_val<uint32_t>(1), dyn = get_arg_val<uint32_t>(2);
    const uint32_t scale = get_arg_val<uint32_t>(3), tau = get_arg_val<uint32_t>(4), mode = get_arg_val<uint32_t>(5);
    const uint32_t thr = get_arg_val<uint32_t>(6), thr2 = get_arg_val<uint32_t>(7);
    const uint32_t inv_hw = get_arg_val<uint32_t>(8), eps = get_arg_val<uint32_t>(9), third = get_arg_val<uint32_t>(10);

    compute_kernel_hw_startup(c_X, c_X, c_Y);
    cb_wait_front(c_SC, 1);
    cb_wait_front(c_X, nt);
    if (conv) {
        cb_wait_front(c_H, 3 * nt);
        cb_wait_front(c_T, 4 * nt);
        if (dyn) {  // DYN / 3 (the dynamic conv's normalizer) -> c_D3
            cb_wait_front(c_D, 4);
            for (uint32_t l = 0; l < 4; ++l) {
                tile_regs_acquire();
                copy_to_dst(c_D, l, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, third);
                done();
                pack_to(0, c_D3);
                tile_regs_release();
            }
            cb_pop_front(c_D, 4);
            cb_wait_front(c_D3, 4);
        }
        for (uint32_t d = 0; d < nt; ++d) {
            tile_regs_acquire();
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t tap = l == 0 ? 0 : 1;
                if (dyn) {
                    reconfig_data_format(c_T, c_D3);
                    add_bcast_cols_init(c_T, c_D3);
                    add_tiles_bcast_cols(c_T, c_D3, l * nt + d, l, tap);
                } else {
                    copy_to_dst(c_T, l * nt + d, tap);
                }
                if (l == 0) {
                    copy_to_dst(c_X, d, 2);
                } else {
                    copy_to_dst(c_H, (l - 1) * nt + d, 2);
                }
                mul_binary_tile_init();
                mul_binary_tile(tap, 2, tap);
                if (l > 0) {
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
            }
            done();
            pack_to(0, c_Y);
            tile_regs_release();
        }
        cb_pop_front(c_H, 3 * nt);
        cb_pop_front(c_T, 4 * nt);
        if (dyn) cb_pop_front(c_D3, 4);
    } else {
        if (scale) cb_wait_front(c_S, nt);
        for (uint32_t d = 0; d < nt; ++d) {
            tile_regs_acquire();
            if (scale) {
                reconfig_data_format(c_X, c_S);
                mul_init(c_X, c_S, false);
                mul_tiles(c_X, c_S, d, d, 0);
            } else {
                copy_to_dst(c_X, d, 0);
            }
            done();
            pack_to(0, c_Y);
            tile_regs_release();
        }
        if (scale) cb_pop_front(c_S, nt);
    }
    cb_pop_front(c_X, nt);
    cb_wait_front(c_Y, nt);
    // Y^2
    for (uint32_t d = 0; d < nt; ++d) {
        tile_regs_acquire();
        reconfig_data_format(c_Y, c_Y);
        mul_init(c_Y, c_Y, false);
        mul_tiles(c_Y, c_Y, d, d, 0);
        done();
        pack_to(0, c_SQ);
        tile_regs_release();
    }
    cb_wait_front(c_SQ, nt);
    // ss = rowsum(Y^2) -> factor
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);  // the reduction accumulates into dst: start from zero
    reconfig_data_format(c_SQ, c_SC);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_SQ, c_SC, c_F);
    for (uint32_t d = 0; d < nt; ++d) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_SQ, c_SC, d, 0, 0);
    }
    reduce_uninit();
    binop_with_scalar_tile_init();
    if (mode == 0) {
        unary_max_tile_init();
        unary_max_tile(0, thr2);
        rsqrt_tile_init();
        rsqrt_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, thr);
    } else {
        mul_unary_tile(0, inv_hw);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
    }
    done();
    pack_to(0, c_F);
    tile_regs_release();
    cb_pop_front(c_SQ, nt);
    uint32_t cf = c_F;
    cb_wait_front(c_F, 1);
    if (tau) {  // f * tau (tau: scalar tile, value at [0, 0])
        cb_wait_front(c_TAU, 1);
        tile_regs_acquire();
        reconfig_data_format(c_F, c_TAU);
        mul_bcast_scalar_init(c_F, c_TAU);
        mul_tiles_bcast_scalar(c_F, c_TAU, 0, 0, 0);
        done();
        pack_to(0, c_F2);
        tile_regs_release();
        cb_pop_front(c_TAU, 1);
        cb_pop_front(c_F, 1);
        cb_wait_front(c_F2, 1);
        cf = c_F2;
    }
    for (uint32_t d = 0; d < nt; ++d) {
        tile_regs_acquire();
        reconfig_data_format(c_Y, cf);
        mul_bcast_cols_init(c_Y, cf);
        mul_tiles_bcast_cols(c_Y, cf, d, 0, 0);
        done();
        pack_to(0, c_O);
        tile_regs_release();
    }
    cb_pop_front(c_Y, nt);
    cb_pop_front(cf, 1);
    cb_pop_front(c_SC, 1);
}
