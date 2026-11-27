// Fused GDN (post-conv) — compute. Per value head h, T tokens (rows of every tile = tokens):
//   Kn = K * rsqrt(rowsum(K^2) + eps)                      (l2 norm; q needs none: the output norm removes it)
//   KT = transpose(Kn)
//   for t: D = ((V - g_t * (Kn@S)) * B) * E_t; S = g_t * S + KT@D (one pass over S; snapshot packed from it);
//          O += (Q@S) * E_t                                (g_t = exp(g) of token t: one scalar per head)
//   OUT = O * rsqrt(rowmean(O^2) + eps) * w * silu(Z)       (gated RMSNorm, bf16 out)
// CT: 0 T, 1 HPC, 2 SEG (rows per lane: each lane of SEG rows starts from its own state, read from c_SIN),
//     3 TG (rows of this core: lanes r0/SEG ..); RT: 0 r0

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
                   c_O0 = 10, c_O1 = 11, c_Z = 12, c_Kn = 13, c_T1 = 14, c_R = 15, c_Sout = 16, c_Out = 17, c_Qn = 18, c_SIN = 19;
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
    constexpr uint32_t SEG = get_compile_time_arg_val(2);
    constexpr uint32_t TG = get_compile_time_arg_val(3);
    const uint32_t r0 = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(c_K, c_SIN, c_D);
    cb_wait_front(c_C, C_E + TG);  // E tiles of this core's rows only (index tl)

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

        cb_wait_front(c_G, TG);
        cb_wait_front(c_B, 1);
        cb_wait_front(c_V, 4);

        for (uint32_t tl = 0; tl < TG; ++tl) {
            const uint32_t t = r0 + tl;  // row (E_t constant, lane boundaries); tl: index of this core's rows
            // a lane's first token reads the lane's initial state from c_SIN (c_S0 / c_S1 are empty then)
            const bool first = (t % SEG) == 0;
            if (first) {
                s_cur = c_SIN;
                s_nxt = c_S0;
                cb_wait_front(c_SIN, 16);
            }
            const bool snap = SNAP_ALL || t % SEG == SEG - 1;

            // 1. D = ((V - g * (Kn @ S)) * B) * E_t        (Kn @ (g S) = g (Kn @ S): g is one scalar per head)
            for (uint32_t j = 0; j < 4; ++j) {
                tile_regs_acquire();
                reconfig_data_format(s_cur, c_Kn);
                matmul_init(c_Kn, s_cur);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(c_Kn, s_cur, i, i * 4 + j, 0);
                }
                reconfig_data_format_srca(c_G);
                copy_init(c_G);
                copy_tile(c_G, tl, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                reconfig_data_format_srca(c_V);
                sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V);
                sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V, j, 0);
                reconfig_data_format_srcb(c_B);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B, 0, 0);
                reconfig_data_format_srcb(c_C);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_E + tl, 0);
                pack_one(c_D);
            }
            cb_wait_front(c_D, 4);

            // 2. S_new = g * S + KT @ D in one pass over the 16 state tiles; the snapshot after token t is packed
            //    from the same registers (all tokens for speculative verify; the lane's last one for prefill)
            for (uint32_t i = 0; i < 4; ++i) {
                for (uint32_t j = 0; j < 4; ++j) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(s_cur);
                    copy_init(s_cur);
                    copy_tile(s_cur, i * 4 + j, 0);
                    reconfig_data_format_srca(c_G);
                    copy_init(c_G);
                    copy_tile(c_G, tl, 1);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 1, 0);
                    reconfig_data_format(c_D, c_KT);
                    matmul_init(c_KT, c_D);
                    matmul_tiles(c_KT, c_D, i, j, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(s_nxt, 1);
                    pack_reconfig_data_format(s_nxt);
                    pack_tile(0, s_nxt);
                    cb_push_back(s_nxt, 1);
                    if (snap) {
                        cb_reserve_back(c_Sout, 1);
                        pack_reconfig_data_format(c_Sout);
                        pack_tile(0, c_Sout);
                        cb_push_back(c_Sout, 1);
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(s_cur, 16);
            cb_pop_front(c_D, 4);
            s_cur = s_nxt;
            s_nxt = (s_nxt == c_S0) ? c_S1 : c_S0;
            cb_wait_front(s_cur, 16);

            // 4. O += (Q @ S) * E_t
            if (tl > 0) {
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
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_E + tl, 0);
                if (tl > 0) {
                    reconfig_data_format_srcb(o_cur);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur, j, 0);
                }
                pack_one(o_nxt);
            }
            if (tl > 0) {
                cb_pop_front(o_cur, 4);
            }
            {
                uint32_t tmp = o_cur;
                o_cur = o_nxt;
                o_nxt = tmp;
            }
            if (t % SEG == SEG - 1) {  // end of the lane: free its state (s_cur alternates c_S0 / c_S1 per token)
                cb_pop_front(s_cur, 16);
            }
        }

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
        cb_pop_front(c_G, TG);
        cb_pop_front(c_B, 1);
        cb_pop_front(c_V, 4);
        cb_pop_front(c_Qn, 4);
    }
    cb_pop_front(c_C, C_E + TG);
}
