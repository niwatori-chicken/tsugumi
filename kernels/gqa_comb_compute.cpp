// GQA split-K attention — combine compute:
//   M = max_s m_s; w_s = exp(m_s - M); L = sum_s l_s w_s; out = sum_s O_s * (w_s / L)   (bf16 out)
// CT: 0 DT (value tiles of this core), 1 S (CB capacity); NA (non-empty splits) comes from the reader
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

constexpr uint32_t c_MS = 0, c_LS = 1, c_OS = 2, c_MX = 3, c_W = 4, c_INV = 5, c_W2 = 6, c_ACC = 7, c_OUT = 8, c_INFO = 9;

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
    constexpr uint32_t HDT = get_compile_time_arg_val(0);
    constexpr uint32_t SC = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(c_OS, c_W2, c_ACC);
    cb_wait_front(c_INFO, 1);
    const uint32_t S = read_tile_value(c_INFO, 0, 0);
    cb_pop_front(c_INFO, 1);
    cb_wait_front(c_MS, SC);
    cb_wait_front(c_LS, SC);

    tile_regs_acquire();
    copy_to_dst(c_MS, 0, 0);
    for (uint32_t s = 1; s < S; ++s) {
        copy_to_dst(c_MS, s, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0);
    }
    done();
    pack_to(0, c_MX);
    tile_regs_release();
    cb_wait_front(c_MX, 1);

    for (uint32_t s = 0; s < S; ++s) {
        tile_regs_acquire();
        copy_to_dst(c_MS, s, 0);
        copy_to_dst(c_MX, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        done();
        pack_to(0, c_W);
        tile_regs_release();
    }
    cb_wait_front(c_W, S);

    tile_regs_acquire();
    copy_to_dst(c_LS, 0, 0);
    copy_to_dst(c_W, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    for (uint32_t s = 1; s < S; ++s) {
        copy_to_dst(c_LS, s, 1);
        copy_to_dst(c_W, s, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    recip_tile_init();
    recip_tile(0);
    done();
    pack_to(0, c_INV);
    tile_regs_release();
    cb_wait_front(c_INV, 1);

    for (uint32_t s = 0; s < S; ++s) {
        tile_regs_acquire();
        copy_to_dst(c_W, s, 0);
        copy_to_dst(c_INV, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        done();
        pack_to(0, c_W2);
        tile_regs_release();
    }
    cb_wait_front(c_W2, S);

    for (uint32_t s = 0; s < S; ++s) {
        cb_wait_front(c_OS, HDT);
        if (s > 0) {
            cb_wait_front(c_ACC, HDT);
        }
        const uint32_t out_cb = (s == S - 1) ? c_OUT : c_ACC;
        for (uint32_t d = 0; d < HDT; ++d) {
            tile_regs_acquire();
            reconfig_data_format(c_OS, c_W2);
            mul_bcast_cols_init(c_OS, c_W2);
            mul_tiles_bcast_cols(c_OS, c_W2, d, s, 1);
            uint32_t res = 1;
            if (s > 0) {
                copy_to_dst(c_ACC, d, 0);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                res = 0;
            }
            done();
            pack_to(res, out_cb);
            tile_regs_release();
        }
        cb_pop_front(c_OS, HDT);
        if (s > 0) {
            cb_pop_front(c_ACC, HDT);
        }
    }
    cb_pop_front(c_MS, SC);
    cb_pop_front(c_LS, SC);
    cb_pop_front(c_MX, 1);
    cb_pop_front(c_W, S);
    cb_pop_front(c_INV, 1);
    cb_pop_front(c_W2, S);
}
