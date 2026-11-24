// Inkling decode: short conv + residual — compute. Per tile:
//   A = HIST + E0 X;  y = ONES0 (A * TAPS) [+ B];  HIST' = SHIFT A
// out tiles -> CB 16, new history tiles -> CB 17.  RT: 0 NT, 1 use_b
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_X = 0, c_H = 1, c_T = 2, c_B = 3, c_C = 4, c_A = 5, c_AT = 6, c_O = 16, c_HN = 17;

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
    const uint32_t NT = get_arg_val<uint32_t>(0), use_b = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(c_C, c_X, c_A);
    cb_wait_front(c_C, 3);
    cb_wait_front(c_X, NT);
    cb_wait_front(c_H, NT);
    cb_wait_front(c_T, NT);
    if (use_b) cb_wait_front(c_B, NT);
    for (uint32_t j = 0; j < NT; ++j) {
        // A = HIST + E0 X  (E0 X: row 0 of X only)
        tile_regs_acquire();
        copy_to_dst(c_H, j, 0);
        reconfig_data_format(c_C, c_X);
        matmul_init(c_C, c_X, 0);
        matmul_tiles(c_C, c_X, 0, j, 0);
        done();
        pack_to(0, c_A);
        tile_regs_release();
        cb_wait_front(c_A, 1);
        // A * TAPS
        tile_regs_acquire();
        reconfig_data_format(c_A, c_T);
        mul_init(c_A, c_T, false);
        mul_tiles(c_A, c_T, 0, j, 0);
        done();
        pack_to(0, c_AT);
        tile_regs_release();
        cb_wait_front(c_AT, 1);
        // y = ONES0 (A * TAPS) [+ B]
        tile_regs_acquire();
        if (use_b) {
            copy_to_dst(c_B, j, 0);
        } else {
            fill_tile_init();
            fill_tile(0, 0.0f);
        }
        reconfig_data_format(c_C, c_AT);
        matmul_init(c_C, c_AT, 0);
        matmul_tiles(c_C, c_AT, 1, 0, 0);
        done();
        pack_to(0, c_O);
        tile_regs_release();
        cb_pop_front(c_AT, 1);
        // HIST' = SHIFT A
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        reconfig_data_format(c_C, c_A);
        matmul_init(c_C, c_A, 0);
        matmul_tiles(c_C, c_A, 2, 0, 0);
        done();
        pack_to(0, c_HN);
        tile_regs_release();
        cb_pop_front(c_A, 1);
    }
    cb_pop_front(c_C, 3);
    cb_pop_front(c_X, NT);
    cb_pop_front(c_H, NT);
    cb_pop_front(c_T, NT);
    if (use_b) cb_pop_front(c_B, NT);
}
