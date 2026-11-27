// Row gather — compute: OUT = MA @ XA + MB @ XB per column tile.  RT: 0 c0, 1 c1
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t c1 = get_arg_val<uint32_t>(1);
    constexpr uint32_t c_X = 0, c_M = 1, c_OUT = 2;
    compute_kernel_hw_startup(c_M, c_X, c_OUT);
    matmul_init(c_M, c_X, 0);
    cb_wait_front(c_M, 2);
    for (uint32_t c = c0; c < c1; ++c) {
        cb_wait_front(c_X, 2);
        tile_regs_acquire();
        matmul_tiles(c_M, c_X, 0, 0, 0);
        matmul_tiles(c_M, c_X, 1, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(c_OUT, 1);
        pack_tile(0, c_OUT);
        cb_push_back(c_OUT, 1);
        tile_regs_release();
        cb_pop_front(c_X, 2);
    }
    cb_pop_front(c_M, 2);
}
