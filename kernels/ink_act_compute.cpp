// Inkling decode MoE: act = tanh(g / gc) * sigmoid(g) * tanh(u / uc) (the softcaps' gc * uc are in the routing
// weights) -> CB 16.  RT: 0 n, 1 1/gc, 2 1/uc (float bits)
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0), igc = get_arg_val<uint32_t>(1), iuc = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(0, 0, 16);
    cb_wait_front(0, n);
    cb_wait_front(1, n);
    for (uint32_t i = 0; i < n; ++i) {  // n = 1: the dense tiles of row 0s
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, i, 0);
        copy_tile(0, i, 1);
        reconfig_data_format_srca(1);
        copy_init(1);
        copy_tile(1, i, 2);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, igc);
        mul_unary_tile(2, iuc);
        tanh_tile_init();
        tanh_tile(0);
        tanh_tile(2);
        sigmoid_tile_init();
        sigmoid_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(16, 1);
        pack_tile(0, 16);
        cb_push_back(16, 1);
        tile_regs_release();
    }
    cb_pop_front(0, n);
    cb_pop_front(1, n);
}
