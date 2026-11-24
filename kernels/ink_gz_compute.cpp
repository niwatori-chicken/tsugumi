// Inkling message gate, phase A — compute: z_part = V @ D (row 0 of RT tiles), ss_part = rowsum(V^2) (column 0),
// to CB 16 (RT tiles z, then 1 tile ss; float32).  RT: 0 KT, 1 RT
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

void kernel_main() {
    const uint32_t KT = get_arg_val<uint32_t>(0), RT = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0, 1, 16);
    cb_wait_front(0, KT);
    cb_wait_front(1, KT * RT);
    cb_wait_front(2, 1);
    for (uint32_t r = 0; r < RT; ++r) {
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        reconfig_data_format(0, 1);
        matmul_init(0, 1, 0);
        for (uint32_t j = 0; j < KT; ++j) matmul_tiles(0, 1, j, j * RT + r, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, 16);
        tile_regs_release();
    }
    for (uint32_t j = 0; j < KT; ++j) {  // V^2 (bf16) -> CB 3
        tile_regs_acquire();
        reconfig_data_format(0, 0);
        mul_init(0, 0, false);
        mul_tiles(0, 0, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, 3);
        tile_regs_release();
    }
    cb_wait_front(3, KT);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(3, 2);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(3, 2, 16);
    for (uint32_t j = 0; j < KT; ++j) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(3, 2, j, 0, 0);
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, 16);
    tile_regs_release();
    cb_pop_front(3, KT);
    cb_pop_front(0, KT);
    cb_pop_front(1, KT * RT);
    cb_pop_front(2, 1);
}
