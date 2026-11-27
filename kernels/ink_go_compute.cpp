// Inkling message gate, phase B — compute:
//   z1 = sum of the h cores' z parts, z2 = the m cores'; ssh, ssm likewise
//   z = z1 * rsqrt(ssh / H + eps) + z2 * rsqrt(ssm / H + eps)      (rms norms of h and m folded into z)
//   f = tanh(z @ F), g = tanh(z @ G) for this core's columns;  out = h - f h + g m  -> CB 16 (bf16)
// RT: 0 NC, 1 RT, 2 CPT, 3 inv_H (float bits), 4 eps (float bits)
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

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
    const uint32_t NC = get_arg_val<uint32_t>(0), RT = get_arg_val<uint32_t>(1), CPT = get_arg_val<uint32_t>(2);
    const uint32_t inv_h = get_arg_val<uint32_t>(3), eps = get_arg_val<uint32_t>(4);
    constexpr uint32_t c_P = 0, c_F = 1, c_H = 2, c_M = 3, c_S = 4, c_R = 5, c_Z = 6, c_FG = 7, c_O = 16;
    compute_kernel_hw_startup(c_P, c_P, c_S);
    cb_wait_front(c_P, NC * (RT + 1));
    const uint32_t half = NC / 2;
    // sums: z1 (RT tiles), z2 (RT), then ssh, ssm -> c_S
    for (uint32_t part = 0; part < 2; ++part) {
        for (uint32_t i = 0; i <= RT; ++i) {
            tile_regs_acquire();
            copy_to_dst(c_P, (part * half) * (RT + 1) + i, 0);
            for (uint32_t cc = 1; cc < half; ++cc) {
                copy_to_dst(c_P, (part * half + cc) * (RT + 1) + i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            done();
            pack_to(0, c_S);
            tile_regs_release();
        }
    }
    cb_pop_front(c_P, NC * (RT + 1));
    cb_wait_front(c_S, 2 * (RT + 1));
    // rh, rm = rsqrt(ss / H + eps)
    for (uint32_t part = 0; part < 2; ++part) {
        tile_regs_acquire();
        copy_to_dst(c_S, part * (RT + 1) + RT, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_h);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        done();
        pack_to(0, c_R);
        tile_regs_release();
    }
    cb_wait_front(c_R, 2);
    // z = z1 * rh + z2 * rm  (bf16, the matmul input)
    for (uint32_t k = 0; k < RT; ++k) {
        tile_regs_acquire();
        reconfig_data_format(c_S, c_R);
        mul_bcast_scalar_init(c_S, c_R);
        mul_tiles_bcast_scalar(c_S, c_R, k, 0, 0);
        mul_tiles_bcast_scalar(c_S, c_R, (RT + 1) + k, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        done();
        pack_to(0, c_Z);
        tile_regs_release();
    }
    cb_pop_front(c_S, 2 * (RT + 1));
    cb_pop_front(c_R, 2);
    cb_wait_front(c_Z, RT);
    cb_wait_front(c_F, 2 * RT * CPT);
    cb_wait_front(c_H, CPT);
    cb_wait_front(c_M, CPT);
    for (uint32_t j = 0; j < CPT; ++j) {
        for (uint32_t g = 0; g < 2; ++g) {  // forget / gain pre-activations -> c_FG
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, 0.0f);
            reconfig_data_format(c_Z, c_F);
            matmul_init(c_Z, c_F, 0);
            for (uint32_t k = 0; k < RT; ++k) matmul_tiles(c_Z, c_F, k, (g * RT + k) * CPT + j, 0);
            done();
            pack_to(0, c_FG);
            tile_regs_release();
        }
        cb_wait_front(c_FG, 2);
        tile_regs_acquire();
        copy_to_dst(c_FG, 0, 0);
        copy_to_dst(c_FG, 1, 1);
        tanh_tile_init();
        tanh_tile(0);
        tanh_tile(1);
        copy_to_dst(c_H, j, 2);
        copy_to_dst(c_M, j, 3);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);  // f h
        mul_binary_tile(1, 3, 1);  // g m
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);  // h - f h
        add_binary_tile_init();
        add_binary_tile(2, 1, 2);
        done();
        pack_to(2, c_O);
        tile_regs_release();
        cb_pop_front(c_FG, 2);
    }
    cb_pop_front(c_Z, RT);
    cb_pop_front(c_F, 2 * RT * CPT);
    cb_pop_front(c_H, CPT);
    cb_pop_front(c_M, CPT);
}
