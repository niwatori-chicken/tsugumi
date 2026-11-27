// Grouped (MoE) GEMM — writer: per unit and row group (RT row tiles computed, the first rr <= RT of them real), per
// column block nb (step CT) the rr * CT tiles (r, nb + c) in the compute's order, one barrier per block.
// SPLIT: columns >= Nt/2 go to a second tensor Y2 (gate | up halves of one matmul into two [R, N/2] tensors).
// CT: 0 Nt, 1 NN, 2 CT, 3 RT, 4 DBG (1: skip the writes), 5 SPLIT, 6 U (unit slots per core), accessors Y, Y2.
// RT: 0 addr Y, 1 n units, then U x (e, r0, nr, n0), then addr Y2
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t NN = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);
    constexpr uint32_t DBG = get_compile_time_arg_val(4);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(5);
    constexpr uint32_t U = get_compile_time_arg_val(6);
    constexpr auto aY = TensorAccessorArgs<7>();
    constexpr auto aY2 = TensorAccessorArgs<aY.next_compile_time_args_offset()>();
    const auto sY = TensorAccessor(aY, get_arg_val<uint32_t>(0));
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const auto sY2 = TensorAccessor(aY2, get_arg_val<uint32_t>(2 + 4 * U));
    constexpr uint32_t Nh = SPLIT ? Nt / 2 : Nt;  // row stride (tiles) of the output tensor(s)
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer co(2);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t r0 = get_arg_val<uint32_t>(3 + 4 * u);
        const uint32_t nr = get_arg_val<uint32_t>(4 + 4 * u);
        const uint32_t n0 = get_arg_val<uint32_t>(5 + 4 * u);
        for (uint32_t g = 0; g < nr; g += RT) {
            const uint32_t rr = nr - g < RT ? nr - g : RT;
            for (uint32_t nb = 0; nb < NN; nb += CT) {
                co.wait_front(RT * CT);
                uint32_t j = 0;
                for (uint32_t r = 0; r < (DBG ? 0 : rr); ++r) {
                    for (uint32_t c = 0; c < CT; ++c, ++j) {
                        const uint32_t col = n0 + nb + c;
                        const uint32_t row = r0 + g + r;
                        if (SPLIT && col >= Nh) {
                            noc.async_write(co, sY2, TB, {.offset_bytes = j * TB}, {.page_id = row * Nh + col - Nh});
                        } else {
                            noc.async_write(co, sY, TB, {.offset_bytes = j * TB}, {.page_id = row * Nh + col});
                        }
                    }
                }
                noc.async_write_barrier();
                co.pop_front(RT * CT);
            }
        }
    }
}
