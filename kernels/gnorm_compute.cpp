// Gated RMSNorm on the head-major GDN output (large-T prefill) — compute. Per unit (32 rows of one head):
//   OUT_j = O_j * rsqrt(rowmean(O^2) + eps) * w_j * silu(Z_j)    (j = 0..3: the head's 128 columns)
// Same arithmetic as the decode kernel (gdn_fused_compute.cpp). RT: 0 u0, 1 u1
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t c_O = 0, c_Z = 1, c_C = 2, c_T1 = 3, c_R = 4, c_Out = 5;
constexpr uint32_t C_ONES128 = 0, C_EPS = 1, C_W = 2;

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
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(c_O, c_C, c_T1);
    cb_wait_front(c_C, 6);
    for (uint32_t u = u0; u < u1; ++u) {
        cb_wait_front(c_O, 4);
        // R = rsqrt(sum_j (O_j*O_j) @ ones/128 + eps): every column of row t = the row's value
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
            reconfig_data_format(c_O, c_O);
            mul_init(c_O, c_O);
            mul_tiles(c_O, c_O, j, j, 0);
            pack_one(c_T1);
        }
        cb_wait_front(c_T1, 4);
        tile_regs_acquire();
        reconfig_data_format(c_T1, c_C);
        matmul_init(c_T1, c_C);
        for (uint32_t j = 0; j < 4; ++j) {
            matmul_tiles(c_T1, c_C, j, C_ONES128, 0);
        }
        reconfig_data_format_srcb(c_C);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_C, C_EPS, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_one(c_R);
        cb_pop_front(c_T1, 4);
        cb_wait_front(c_R, 1);
        cb_wait_front(c_Z, 4);
        for (uint32_t j = 0; j < 4; ++j) {
            tile_regs_acquire();
            reconfig_data_format(c_O, c_R);
            mul_init(c_O, c_R);
            mul_tiles(c_O, c_R, j, 0, 0);
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
            pack_one(c_Out);
        }
        cb_pop_front(c_R, 1);
        cb_pop_front(c_O, 4);
        cb_pop_front(c_Z, 4);
    }
    cb_pop_front(c_C, 6);
}
