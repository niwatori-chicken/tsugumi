// Inkling decode: attention output gate — compute on the dense tiles: out = o * sigmoid(g) -> CB 16.
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    compute_kernel_hw_startup(0, 0, 16);
    cb_wait_front(0, 1);
    cb_wait_front(1, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(0);
    copy_init(0);
    copy_tile(0, 0, 0);
    reconfig_data_format_srca(1);
    copy_init(1);
    copy_tile(1, 0, 1);
    sigmoid_tile_init();
    sigmoid_tile(0);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(16, 1);
    pack_tile(0, 16);
    cb_push_back(16, 1);
    tile_regs_release();
    cb_pop_front(0, 1);
    cb_pop_front(1, 1);
}
