// GDN recurrence (fused, multi-token) — compute. One head at a time, T tokens sequentially:
//   S <- S * exp(g_t)                          (SFPU fp32 multiply, G_t is a full tile of exp(g_t))
//   D <- ((V - K @ S) * B) * E_t               (rows = tokens; E_t keeps only row t)
//   S <- S + K^T @ D                           (only token t contributes)
//   O <- O + (Q @ S) * E_t
// State S: 4x4 fp32 tiles (row block i = key dims, col block j = value dims).
// CT args: 0 T, 1 heads_per_core.

#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"

constexpr uint32_t c_S0 = 0, c_S1 = 1, c_Q = 2, c_K = 3, c_KT = 4, c_V = 5, c_B = 6, c_G = 7, c_E = 8, c_D = 9,
                   c_O0 = 10, c_O1 = 11, c_Sout = 16, c_Oout = 17;

FORCE_INLINE void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(c_K, c_S0, c_D);
    cb_wait_front(c_E, T);

    for (uint32_t hh = 0; hh < HPC; ++hh) {
        uint32_t s_cur = c_S0, s_nxt = c_S1;
        uint32_t o_cur = c_O0, o_nxt = c_O1;
        cb_wait_front(c_S0, 16);
        cb_wait_front(c_G, T);
        cb_wait_front(c_K, 4);
        cb_wait_front(c_V, 4);
        cb_wait_front(c_B, 1);
        cb_wait_front(c_KT, 4);
        cb_wait_front(c_Q, 4);

        for (uint32_t t = 0; t < T; ++t) {
            // 1. decay: S <- S * G_t
            for (uint32_t idx = 0; idx < 16; ++idx) {
                tile_regs_acquire();
                copy_init(s_cur);
                copy_tile(s_cur, idx, 0);
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

            // 2. D_j = ((V_j - sum_i K_i @ S_ij) * B) * E_t
            for (uint32_t j = 0; j < 4; ++j) {
                tile_regs_acquire();
                matmul_init(c_K, s_cur);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(c_K, s_cur, i, i * 4 + j, 0);
                }
                sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V);
                sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(c_V, j, 0);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_B, 0, 0);
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_E);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_E, t, 0);
                pack_one(c_D);
            }
            cb_wait_front(c_D, 4);

            // 3. S_ij <- S_ij + KT_i @ D_j
            for (uint32_t i = 0; i < 4; ++i) {
                for (uint32_t j = 0; j < 4; ++j) {
                    tile_regs_acquire();
                    copy_init(s_cur);
                    copy_tile(s_cur, i * 4 + j, 0);
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

            // 4. O_j <- O_j + (sum_i Q_i @ S_ij) * E_t
            const bool last = (t == T - 1);
            const uint32_t o_dst = last ? c_Oout : o_nxt;
            if (t > 0) {
                cb_wait_front(o_cur, 4);
            }
            for (uint32_t j = 0; j < 4; ++j) {
                tile_regs_acquire();
                matmul_init(c_Q, s_cur);
                for (uint32_t i = 0; i < 4; ++i) {
                    matmul_tiles(c_Q, s_cur, i, i * 4 + j, 0);
                }
                mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_E);
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_E, t, 0);
                if (t > 0) {
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(o_cur, j, 0);
                }
                pack_one(o_dst);
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

        // final state out
        for (uint32_t idx = 0; idx < 16; ++idx) {
            tile_regs_acquire();
            copy_init(s_cur);
            copy_tile(s_cur, idx, 0);
            pack_one(c_Sout);
        }
        cb_pop_front(s_cur, 16);
        cb_pop_front(c_G, T);
        cb_pop_front(c_K, 4);
        cb_pop_front(c_V, 4);
        cb_pop_front(c_B, 1);
        cb_pop_front(c_KT, 4);
        cb_pop_front(c_Q, 4);
    }
    cb_pop_front(c_E, T);
}
