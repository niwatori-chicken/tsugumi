// Causal depthwise conv (kernel 4) + SiLU — compute. For output tile i of a column tile (prev = tile i-1):
//   y = silu( sum_j P_j (prev * tap_j) + Q_j (cur * tap_j) )
// (row-broadcast scaling by tap_j commutes with the row-shift matrices P_j / Q_j).
// CT: 0 NT, 1 CTL, 2 RPS.  RT: 0 u0, 1 u1
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_X = 0, c_TAP = 1, c_PQ = 2, c_S0 = 3, c_OUT = 4, c_S1 = 5;  // S0/S1: ping-pong (no wrap)

FORCE_INLINE void scale4(uint32_t c_S) {
    // c_S <- cur * tap_j (j = 0..3), cur = front of c_X
    cb_wait_front(c_X, 1);
    tile_regs_acquire();
    reconfig_data_format(c_X, c_TAP);
    mul_bcast_rows_init(c_X, c_TAP);
    for (uint32_t j = 0; j < 4; ++j) {
        mul_tiles_bcast_rows(c_X, c_TAP, 0, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(c_S, 4);
    pack_reconfig_data_format(c_S);
    for (uint32_t j = 0; j < 4; ++j) {
        pack_tile(j, c_S);
    }
    cb_push_back(c_S, 4);
    tile_regs_release();
    cb_pop_front(c_X, 1);
}

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t CTL = get_compile_time_arg_val(1);
    constexpr uint32_t RPS = get_compile_time_arg_val(2);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(c_X, c_TAP, c_S0);
    cb_wait_front(c_PQ, 8);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t s = u / CTL;
        const uint32_t i0 = s * RPS, i1 = (i0 + RPS < NT) ? i0 + RPS : NT;
        cb_wait_front(c_TAP, 4);
        uint32_t sp = c_S0, sc = c_S1;
        scale4(sp);  // prev
        for (uint32_t i = i0; i < i1; ++i) {
            scale4(sc);  // cur
            cb_wait_front(sp, 4);
            cb_wait_front(sc, 4);
            tile_regs_acquire();
            reconfig_data_format(sp, c_PQ);
            matmul_init(c_PQ, sp, 0);
            for (uint32_t j = 0; j < 3; ++j) {
                matmul_tiles(c_PQ, sp, j, j, 0);  // P_j @ (prev * tap_j)
            }
            for (uint32_t j = 0; j < 4; ++j) {
                matmul_tiles(c_PQ, sc, 4 + j, j, 0);  // Q_j @ (cur * tap_j)
            }
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(c_OUT, 1);
            pack_reconfig_data_format(c_OUT);
            pack_tile(0, c_OUT);
            cb_push_back(c_OUT, 1);
            tile_regs_release();
            cb_pop_front(sp, 4);
            const uint32_t t = sp; sp = sc; sc = t;
        }
        cb_pop_front(sp, 4);
        cb_pop_front(c_TAP, 4);
    }
}
