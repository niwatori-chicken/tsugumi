// GQA split-K attention — compute. Online softmax over this core's key chunks:
//   S = Q K^T (+ mask); m' = max(m, rowmax S); a = exp(m - m'); P = exp(S - m');
//   l' = l a + rowsum P; O' = O a + P V
// Outputs (unnormalized) O [RT x HDT], m [RT], l [RT] (fp32, value in column 0 for m / l).
// CT: 0 RT, 1 CT, 2 HDT
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_Q = 0, c_K = 1, c_V = 2, c_MASK = 3, c_SC = 4, c_INFO = 5, c_S = 6, c_P = 7, c_M0 = 8,
                   c_M1 = 9, c_L0 = 10, c_L1 = 11, c_A = 12, c_O0 = 13, c_O1 = 14, c_FO = 16, c_FM = 17, c_FL = 18,
                   c_CONST = 19;

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

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t HDT = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(c_Q, c_K, c_S);

    cb_wait_front(c_INFO, 1);
    const uint32_t nloc = read_tile_value(c_INFO, 0, 0);
    const uint32_t c0 = read_tile_value(c_INFO, 0, 1);
    const uint32_t P0 = read_tile_value(c_INFO, 0, 2);
    cb_pop_front(c_INFO, 1);
    cb_wait_front(c_SC, 1);
    cb_wait_front(c_CONST, 2);
    if (nloc) cb_wait_front(c_Q, RT * HDT);

    // init m = -1e30, l = 0, O = 0  (m and l live in one CB each, RT tiles per generation; O: RT*HDT)
    for (uint32_t r = 0; r < RT; ++r) {
        tile_regs_acquire();
        copy_to_dst(c_CONST, 0, 0);
        copy_to_dst(c_CONST, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, c_M0);
        pack_to(1, c_L0);
        tile_regs_release();
    }
    for (uint32_t i = 0; i < RT * HDT; ++i) {
        tile_regs_acquire();
        copy_to_dst(c_CONST, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, c_O0);
        tile_regs_release();
    }
    uint32_t m_cur = c_M0, m_nxt = c_M1, l_cur = c_L0, l_nxt = c_L1, o_cur = c_O0, o_nxt = c_O1;

    for (uint32_t i = 0; i < nloc; ++i) {
        const bool masked = (c0 + i + 1) * CT * 32 > P0 + 1;
        cb_wait_front(c_K, CT * HDT);
        if (masked) {
            cb_wait_front(c_MASK, RT * CT);
        }
        // S = (mask) + Q K^T
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t j = 0; j < CT; ++j) {
                tile_regs_acquire();
                if (masked) {
                    copy_to_dst(c_MASK, r * CT + j, 0);
                }
                reconfig_data_format(c_K, c_Q);
                matmul_init(c_Q, c_K, 1);
                for (uint32_t d = 0; d < HDT; ++d) {
                    matmul_tiles(c_Q, c_K, r * HDT + d, j * HDT + d, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_to(0, c_S);
                tile_regs_release();
            }
        }
        cb_pop_front(c_K, CT * HDT);
        if (masked) {
            cb_pop_front(c_MASK, RT * CT);
        }
        cb_wait_front(c_S, RT * CT);
        cb_wait_front(m_cur, RT);
        cb_wait_front(l_cur, RT);
        // m' = max(m, rowmax S)
        for (uint32_t r = 0; r < RT; ++r) {
            tile_regs_acquire();
            reconfig_data_format(c_S, c_SC);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(c_S, c_SC, m_nxt);
            for (uint32_t j = 0; j < CT; ++j) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(c_S, c_SC, r * CT + j, 0, 0);
            }
            reduce_uninit();
            copy_to_dst(m_cur, r, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, m_nxt);
            tile_regs_release();
        }
        // a = exp(m - m') from the stored (unpacked) values so every use sees the same m'
        cb_wait_front(m_nxt, RT);
        for (uint32_t r = 0; r < RT; ++r) {
            tile_regs_acquire();
            copy_to_dst(m_cur, r, 0);
            copy_to_dst(m_nxt, r, 1);
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
            exp_tile_init();
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, c_A);
            tile_regs_release();
        }
        // P = exp(S - m')
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t j = 0; j < CT; ++j) {
                tile_regs_acquire();
                reconfig_data_format(c_S, m_nxt);
                sub_bcast_cols_init(c_S, m_nxt);
                sub_tiles_bcast_cols(c_S, m_nxt, r * CT + j, r, 0);
                exp_tile_init();
                exp_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_to(0, c_P);
                tile_regs_release();
            }
        }
        cb_pop_front(c_S, RT * CT);
        cb_wait_front(c_P, RT * CT);
        cb_wait_front(c_A, RT);
        // l' = l a + rowsum P
        for (uint32_t r = 0; r < RT; ++r) {
            tile_regs_acquire();
            reconfig_data_format(c_SC, c_P);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_P, c_SC, l_nxt);
            for (uint32_t j = 0; j < CT; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(c_P, c_SC, r * CT + j, 0, 0);
            }
            reduce_uninit();
            copy_to_dst(l_cur, r, 1);
            copy_to_dst(c_A, r, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, l_nxt);
            tile_regs_release();
        }
        // O' = O a + P V
        cb_wait_front(c_V, CT * HDT);
        cb_wait_front(o_cur, RT * HDT);
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t d = 0; d < HDT; ++d) {
                tile_regs_acquire();
                reconfig_data_format(o_cur, c_A);
                mul_bcast_cols_init(o_cur, c_A);
                mul_tiles_bcast_cols(o_cur, c_A, r * HDT + d, r, 0);
                reconfig_data_format(c_V, c_P);
                matmul_init(c_P, c_V, 0);
                for (uint32_t j = 0; j < CT; ++j) {
                    matmul_tiles(c_P, c_V, r * CT + j, j * HDT + d, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_to(0, o_nxt);
                tile_regs_release();
            }
        }
        cb_pop_front(c_V, CT * HDT);
        cb_pop_front(c_P, RT * CT);
        cb_pop_front(c_A, RT);
        cb_pop_front(o_cur, RT * HDT);
        cb_pop_front(m_cur, RT);
        cb_pop_front(l_cur, RT);
        uint32_t tmp = m_cur; m_cur = m_nxt; m_nxt = tmp;
        tmp = l_cur; l_cur = l_nxt; l_nxt = tmp;
        tmp = o_cur; o_cur = o_nxt; o_nxt = tmp;
    }

    // final: O, m, l -> writer
    cb_wait_front(o_cur, RT * HDT);
    cb_wait_front(m_cur, RT);
    cb_wait_front(l_cur, RT);
    for (uint32_t r = 0; r < RT; ++r) {
        for (uint32_t d = 0; d < HDT; ++d) {
            tile_regs_acquire();
            copy_to_dst(o_cur, r * HDT + d, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, c_FO);
            tile_regs_release();
        }
    }
    for (uint32_t r = 0; r < RT; ++r) {
        tile_regs_acquire();
        copy_to_dst(m_cur, r, 0);
        copy_to_dst(l_cur, r, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, c_FM);
        pack_to(1, c_FL);
        tile_regs_release();
    }
    cb_pop_front(o_cur, RT * HDT);
    cb_pop_front(m_cur, RT);
    cb_pop_front(l_cur, RT);
    if (nloc) cb_pop_front(c_Q, RT * HDT);
    cb_pop_front(c_SC, 1);
    cb_pop_front(c_CONST, 2);
}
