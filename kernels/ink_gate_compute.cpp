// Inkling message gate in one program — compute (see ink_gate_reader.cpp).
//   A: z_part = V @ D (RT tiles, row 0), ss_part = rowsum(V^2) -> CB 4 (PA, float32: RT + 1 tiles)
//   sums: SEL_h / SEL_m @ DENSE -> z_h, z_m (row 0), ss_h, ss_m ([0][0]);  r = rsqrt(ss / H + eps)
//   z = z_h r_h + z_m r_m -> CB 11 (bf16);  f, g = tanh(z @ [F | G]);  out = h - f h + g m -> CB 16
// RT: 0 KT, 1 RT, 2 CPT, 3 1/H, 4 eps (float bits)
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
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_V = 0, c_D = 1, c_SC = 2, c_VSQ = 3, c_PA = 4, c_F = 5, c_H = 6, c_M = 7, c_DN = 8, c_SEL = 9,
                   c_R = 10, c_Z = 11, c_FG = 12, c_ZT = 13, c_O = 16;

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
    const uint32_t KT = get_arg_val<uint32_t>(0), RT = get_arg_val<uint32_t>(1), CPT = get_arg_val<uint32_t>(2);
    const uint32_t inv_h = get_arg_val<uint32_t>(3), eps = get_arg_val<uint32_t>(4);
    compute_kernel_hw_startup(c_D, c_V, c_PA);
    // ---- phase A
    cb_wait_front(c_V, KT);
    cb_wait_front(c_D, KT * RT);
    cb_wait_front(c_SC, 1);
    for (uint32_t r = 0; r < RT; ++r) {
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        reconfig_data_format(c_D, c_V);
        matmul_init(c_V, c_D, 0);
        for (uint32_t j = 0; j < KT; ++j) matmul_tiles(c_V, c_D, j, j * RT + r, 0);
        done();
        pack_to(0, c_PA);
        tile_regs_release();
    }
    for (uint32_t j = 0; j < KT; ++j) {
        tile_regs_acquire();
        reconfig_data_format(c_V, c_V);
        mul_init(c_V, c_V, false);
        mul_tiles(c_V, c_V, j, j, 0);
        done();
        pack_to(0, c_VSQ);
        tile_regs_release();
    }
    cb_wait_front(c_VSQ, KT);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(c_VSQ, c_SC);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_VSQ, c_SC, c_PA);
    for (uint32_t j = 0; j < KT; ++j) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_VSQ, c_SC, j, 0, 0);
    reduce_uninit();
    done();
    pack_to(0, c_PA);
    tile_regs_release();
    cb_pop_front(c_VSQ, KT);
    cb_pop_front(c_V, KT);
    cb_pop_front(c_D, KT * RT);
    cb_pop_front(c_SC, 1);
    // ---- sums over the cores
    cb_wait_front(c_SEL, 2);
    cb_wait_front(c_DN, RT + 1);
    tile_regs_acquire();  // ss_h, ss_m -> r_h, r_m
    fill_tile_init();
    fill_tile(0, 0.0f);
    fill_tile(1, 0.0f);
    reconfig_data_format(c_DN, c_SEL);
    matmul_init(c_SEL, c_DN, 0);
    matmul_tiles(c_SEL, c_DN, 0, RT, 0);
    matmul_tiles(c_SEL, c_DN, 1, RT, 1);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    mul_unary_tile(1, inv_h);
    add_unary_tile(0, eps);
    add_unary_tile(1, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    rsqrt_tile(1);
    done();
    pack_to(0, c_R);
    pack_to(1, c_R);
    tile_regs_release();
    cb_wait_front(c_R, 2);
    for (uint32_t k = 0; k < RT; ++k) {
        tile_regs_acquire();  // z_h, z_m of column tile k
        fill_tile_init();
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        reconfig_data_format(c_DN, c_SEL);
        matmul_init(c_SEL, c_DN, 0);
        matmul_tiles(c_SEL, c_DN, 0, k, 0);
        matmul_tiles(c_SEL, c_DN, 1, k, 1);
        done();
        pack_to(0, c_ZT);
        pack_to(1, c_ZT);
        tile_regs_release();
        cb_wait_front(c_ZT, 2);
        tile_regs_acquire();  // z = z_h r_h + z_m r_m
        reconfig_data_format(c_ZT, c_R);
        mul_bcast_scalar_init(c_ZT, c_R);
        mul_tiles_bcast_scalar(c_ZT, c_R, 0, 0, 0);
        mul_tiles_bcast_scalar(c_ZT, c_R, 1, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        done();
        pack_to(0, c_Z);
        tile_regs_release();
        cb_pop_front(c_ZT, 2);
    }
    cb_pop_front(c_R, 2);
    cb_pop_front(c_DN, RT + 1);
    cb_pop_front(c_SEL, 2);
    // ---- phase B
    cb_wait_front(c_Z, RT);
    cb_wait_front(c_F, 2 * RT * CPT);
    cb_wait_front(c_H, CPT);
    cb_wait_front(c_M, CPT);
    for (uint32_t j = 0; j < CPT; ++j) {
        for (uint32_t g = 0; g < 2; ++g) {
            tile_regs_acquire();
            fill_tile_init();
            fill_tile(0, 0.0f);
            reconfig_data_format(c_F, c_Z);
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
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(1, 3, 1);
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);
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
