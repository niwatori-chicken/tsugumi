// Grouped (MoE) GEMM — compute: per unit, blocks of RT row tiles x CT column tiles with matmul_block (each unpacked
// W tile serves RT rows), the unit's W slice ([Kt][NN] tiles) held in CB 1, RT row tiles of X in CB 0 (always RT,
// so the CBs wrap evenly: the last group of a unit computes rows beyond it and the writer drops them).
// Output order per row group: for nb (step CT): for r < RT: for c < CT: tile (r, nb + c).
// CT: 0 Kt, 1 NN, 2 CT, 3 RT (RT * CT <= 4 with fp32 dest).  RT: 0 n units, then 4 values per unit (nr = slot 2)
#include <stdint.h>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t NN = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);
    constexpr uint32_t SKIP = get_compile_time_arg_val(4);  // debug: 1 = no matmul (data movement only)
    const uint32_t nu = get_arg_val<uint32_t>(0);
    constexpr uint32_t c_X = 0, c_W = 1, c_OUT = 2;
    compute_kernel_hw_startup(c_W, c_X, c_OUT);  // srcA = in1 (W) format, srcB = in0 (X)
    matmul_block_init(c_X, c_W, 0, CT, RT, Kt);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t nr = get_arg_val<uint32_t>(3 + 4 * u);
        cb_wait_front(c_W, Kt * NN);
        for (uint32_t g = 0; g < nr; g += RT) {
            cb_wait_front(c_X, RT * Kt);
            for (uint32_t nb = 0; nb < NN; nb += CT) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < (SKIP ? 0 : Kt); ++k) {
                    matmul_block(c_X, c_W, k, k * NN + nb, 0, false, CT, RT, Kt);
                }
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(c_OUT, RT * CT);
                for (uint32_t j = 0; j < RT * CT; ++j) {
                    pack_tile(j, c_OUT);
                }
                cb_push_back(c_OUT, RT * CT);
                tile_regs_release();
            }
            cb_pop_front(c_X, RT * Kt);
        }
        cb_pop_front(c_W, Kt * NN);
    }
}
