// Inkling decode MoE routing — compute: s = sigmoid(logits) -> CB 2, s + bias -> CB 3 (float32).  RT: 0 tiles
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    const uint32_t nt = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 0, 2);
    cb_wait_front(0, nt);
    cb_wait_front(1, nt);
    for (uint32_t t = 0; t < nt; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(0);
        copy_init(0);
        copy_tile(0, t, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        reconfig_data_format_srca(1);
        copy_init(1);
        copy_tile(1, t, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(2, 1);
        pack_reconfig_data_format(2);
        pack_tile(0, 2);
        cb_push_back(2, 1);
        cb_reserve_back(3, 1);
        pack_reconfig_data_format(3);
        pack_tile(1, 3);
        cb_push_back(3, 1);
        tile_regs_release();
    }
    cb_pop_front(0, nt);
    cb_pop_front(1, nt);
}
