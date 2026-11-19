// Fused GDN (post-conv) — compute. Per value head h, T tokens (rows of every tile = tokens):
//   Kn = K * rsqrt(rowsum(K^2) + eps)                      (l2 norm; q needs none: the output norm removes it)
//   KT = transpose(Kn)
//   for t: S *= exp(g_t); D = ((V - Kn@S) * B) * E_t; S += KT@D; O += (Q@S) * E_t
//   OUT = O * rsqrt(rowmean(O^2) + eps) * w * silu(Z)       (gated RMSNorm, bf16 out)
// CT: 0 T, 1 HPC

#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"

#ifndef DBG
#define DBG 0
#endif
#ifndef SNAP_ALL
#define SNAP_ALL 1
#endif

constexpr uint32_t c_S0 = 0, c_S1 = 1, c_Q = 2, c_K = 3, c_KT = 4, c_V = 5, c_B = 6, c_G = 7, c_C = 8, c_D = 9,
                   c_O0 = 10, c_O1 = 11, c_Z = 12, c_Kn = 13, c_T1 = 14, c_R = 15, c_Sout = 16, c_Out = 17, c_Qn = 18;
constexpr uint32_t C_ONES = 0, C_ONES128 = 1, C_EPS = 2, C_W = 3, C_X128 = 7, C_EPS128 = 8, C_E = 9;

FORCE_INLINE void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

// R = rsqrt(sum_j (X_j*X_j) @ ones_k + eps)   (every column of row t = the row's value)
FORCE_INLINE void row_rsqrt(uint32_t cb_x, uint32_t ones_idx, uint32_t eps_idx) {
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_x);
                mul_init(cb_x, cb_x);
        mul_tiles(cb_x, cb_x, j, j, 0);
        pack_one(c_T1);
    }
    cb_wait_front(c_T1, 4);
    tile_regs_acquire();
    reconfig_data_format(c_C, c_T1);
                matmul_init(c_T1, c_C);
    for (uint32_t j = 0; j < 4; ++j) {
        matmul_tiles(c_T1, c_C, j, ones_idx, 0);
    }
    reconfig_data_format_srcb(c_C);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, eps_idx, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_one(c_R);
    cb_pop_front(c_T1, 4);
    cb_wait_front(c_R, 1);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(c_K, c_S0, c_D);
    cb_wait_front(c_C, C_E + T);

    for (uint32_t hh = 0; hh < HPC; ++hh) {
        uint32_t s_cur = c_S0, s_nxt = c_S1;
        uint32_t o_cur = c_O0, o_nxt = c_O1;

        // ---- k l2 norm + transpose ----
        cb_wait_front(c_K, 4);
        row_rsqrt(c_K, C_ONES, C_EPS);
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
            reconfig_data_format(c_K, c_R);
                mul_init(c_K, c_R);
            mul_tiles(c_K, c_R, j, 0, 0);
            pack_one(c_Kn);
        }
        cb_pop_front(c_R, 1);
        cb_pop_front(c_K, 4);
        cb_wait_front(c_Kn, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            tile_regs_acquire();
            reconfig_data_format_srca(c_Kn);
                transpose_init(c_Kn);
            transpose_tile(c_Kn, i, 0);
            pack_one(c_KT);
        }
        cb_wait_front(c_KT, 4);

        // ---- q: l2 norm * Dk^-0.5 = rsqrt(128*sumsq + 128*eps) ----
        cb_wait_front(c_Q, 4);
        row_rsqrt(c_Q, C_X128, C_EPS128);
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
            reconfig_data_format(c_Q, c_R);
            mul_init(c_Q, c_R);
            mul_tiles(c_Q, c_R, j, 0, 0);
            pack_one(c_Qn);
        }
        cb_pop_front(c_R, 1);
        cb_pop_front(c_Q, 4);
        cb_wait_front(c_Qn, 4);

        cb_wait_front(c_S0, 16);
        cb_wait_front(c_G, T);
        cb_wait_front(c_B, 1);
        cb_wait_front(c_V, 4);

        for (uint32_t t = 0; t < T; ++t) {
            // 1. decay
            for (uint32_t idx = 0; idx < 16; ++idx) {
                tile_regs_acquire();
                reconfig_data_format_srca(s_cur);
                copy_init(s_cur);
                copy_tile(s_cur, idx, 0);
                reconfig_data_format_srca(c_G);
                copy_init(c_G);
                copy_tile(c_G, t, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                pack_one(s_nxt);
            }
            cb_pop_front(s_cur, 16);
            {
                uint32_t tmp = s_cur;
                s_cur = s_nxt;
                s_nxt = tmp;
            }
            cb_wait_front(s_cur, 16);

            // 2. D
            for (uint32_t j = 0; j < 4; ++j) {
                tile_regs_acquire();
                reconfig_data_format(s_cur, c_Kn);
                matmul_init(c_Kn, s_cur);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(c_Kn, s_cur, i, i * 4 + j, 0);
                }
                reconfig_data_format_srca(c_V);
                sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V);
                sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V, j, 0);
                reconfig_data_format_srcb(c_B);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B, 0, 0);
                reconfig_data_format_srcb(c_C);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_E + t, 0);
                pack_one(c_D);
            }
            cb_wait_front(c_D, 4);

            // 3. S += KT @ D
            for (uint32_t i = 0; i < 4; ++i) {
                for (uint32_t j = 0; j < 4; ++j) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(s_cur);
                copy_init(s_cur);
                    copy_tile(s_cur, i * 4 + j, 0);
                    reconfig_data_format(c_D, c_KT);
                matmul_init(c_KT, c_D);
                    matmul_tiles(c_KT, c_D, i, j, 0);
                    pack_one(s_nxt);
                }
            }
            cb_pop_front(s_cur, 16);
            cb_pop_front(c_D, 4);
            {
                uint32_t tmp = s_cur;
                s_cur = s_nxt;
                s_nxt = tmp;
            }
            cb_wait_front(s_cur, 16);

            // snapshot after token t (all tokens for speculative verify; only the last one for prefill)
            if (SNAP_ALL || t == T - 1)
            for (uint32_t idx = 0; idx < 16; ++idx) {
                tile_regs_acquire();
                reconfig_data_format_srca(s_cur);
                copy_init(s_cur);
                copy_tile(s_cur, idx, 0);
                pack_one(c_Sout);
            }

            // 4. O += (Q @ S) * E_t
            if (t > 0) {
                cb_wait_front(o_cur, 4);
            }
            for (uint32_t j = 0; j < 4; ++j) {
                tile_regs_acquire();
                reconfig_data_format(s_cur, c_Qn);
                matmul_init(c_Qn, s_cur);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(c_Qn, s_cur, i, i * 4 + j, 0);
                }
                reconfig_data_format_srcb(c_C);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_E + t, 0);
                if (t > 0) {
                    reconfig_data_format_srcb(o_cur);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur, j, 0);
                }
                pack_one(o_nxt);
            }
            if (t > 0) {
                cb_pop_front(o_cur, 4);
            }
            {
                uint32_t tmp = o_cur;
                o_cur = o_nxt;
                o_nxt = tmp;
            }
        }

        cb_pop_front(s_cur, 16);

        // ---- gated RMSNorm: OUT_j = O_j * R * w_j * silu(Z_j) ----
        cb_wait_front(o_cur, 4);
        row_rsqrt(o_cur, C_ONES128, C_EPS);
        cb_wait_front(c_Z, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
#if DBG == 1
            reconfig_data_format_srca(c_Kn);
                copy_init(c_Kn);
            copy_tile(c_Kn, j, 0);
#elif DBG == 2
            reconfig_data_format_srca(c_KT);
                copy_init(c_KT);
            copy_tile(c_KT, j, 0);
#elif DBG == 3
            reconfig_data_format_srca(o_cur);
                copy_init(o_cur);
            copy_tile(o_cur, j, 0);
#elif DBG == 4
            reconfig_data_format_srca(c_R);
                copy_init(c_R);
            copy_tile(c_R, 0, 0);
#elif DBG == 5
            reconfig_data_format_srca(c_B);
                copy_init(c_B);
            copy_tile(c_B, 0, 0);
#elif DBG == 6
            reconfig_data_format_srca(c_G);
                copy_init(c_G);
            copy_tile(c_G, 0, 0);
#else
            reconfig_data_format(o_cur, c_R);
                mul_init(o_cur, c_R);
            mul_tiles(o_cur, c_R, j, 0, 0);
            reconfig_data_format_srcb(c_C);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_W + j, 0);
            reconfig_data_format_srca(c_Z);
                copy_init(c_Z);
            copy_tile(c_Z, j, 1);
            silu_tile_init();
            silu_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#endif
            pack_one(c_Out);
        }
        cb_pop_front(c_R, 1);
        cb_pop_front(o_cur, 4);
        cb_pop_front(c_Z, 4);
        cb_pop_front(c_KT, 4);
        cb_pop_front(c_Kn, 4);
        cb_pop_front(c_G, T);
        cb_pop_front(c_B, 1);
        cb_pop_front(c_V, 4);
        cb_pop_front(c_Qn, 4);
    }
    cb_pop_front(c_C, C_E + T);
}
