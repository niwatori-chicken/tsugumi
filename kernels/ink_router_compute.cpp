// Inkling decode MoE router + routing — compute. Every core: partial logits = sum over its K tiles of x @ Wg
// (row 0, float32) -> CB 4. Core 0 then: logits of tile t = sum of the KS partials (CB 5, unpacked to DST in
// float32), s = sigmoid(logits) -> CB 2, s + bias -> CB 3 (the writer selects the top K).
// RT: 0 KTC, 1 is core 0, 2 NT, 3 KS
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
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

FORCE_INLINE void copy_to_dst(uint32_t cb, uint32_t idx, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, idx, dst);
}

void kernel_main() {
    const uint32_t KTC = get_arg_val<uint32_t>(0), route = get_arg_val<uint32_t>(1);
    const uint32_t NT = get_arg_val<uint32_t>(2), KS = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup(1, 0, 4);
    cb_wait_front(0, KTC);
    cb_wait_front(1, KTC);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(1, 0);
    matmul_init(0, 1, 0);
    for (uint32_t j = 0; j < KTC; ++j) matmul_tiles(0, 1, j, j, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, 4);
    tile_regs_release();
    cb_pop_front(0, KTC);
    cb_pop_front(1, KTC);
    if (!route) return;
    cb_wait_front(5, NT * KS);
    cb_wait_front(6, NT);
    for (uint32_t t = 0; t < NT; ++t) {
        tile_regs_acquire();
        copy_to_dst(5, t * KS, 0);
        for (uint32_t k = 1; k < KS; ++k) {
            copy_to_dst(5, t * KS + k, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        sigmoid_tile_init();
        sigmoid_tile(0);
        copy_to_dst(6, t, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, 2);
        pack_to(1, 3);
        tile_regs_release();
    }
    cb_pop_front(5, NT * KS);
    cb_pop_front(6, NT);
}
