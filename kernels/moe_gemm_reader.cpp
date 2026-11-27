// Grouped (MoE) GEMM, large-T prefill — reader. Y[rows of expert e] = X[rows] @ W[e] for a host-built schedule.
// X [1,1,R,K] bf16 TILE (rows sorted by expert, each expert's rows padded to whole row tiles),
// W [1,1,E*K,N] bfp4 TILE (expert e in rows e*K..), Y [1,1,R,N] bf16 TILE.
// A unit (e, r0, nr, n0) = row tiles r0..r0+nr-1 of expert e, output column tiles n0..n0+NN-1: the [K, NN] slice of
// W[e] is read into L1 once and every row tile of the unit is multiplied by it.
// CT: 0 Kt, 1 Nt, 2 NN, 3 U (units per core, padded), 4 W tile bytes, 5 RT (row tiles per block), 6 WPAGE (1: W
// is a row-major tensor with one page per (expert, column slice), see MoEGemm.relayout), accessors X, W.
// RT: 0 addr X, 1 addr W, 2 n units, then U x (e, r0, nr, n0), then the row tiles of X
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t NN = get_compile_time_arg_val(2);
    constexpr uint32_t WB = get_compile_time_arg_val(4);
    constexpr uint32_t RT = get_compile_time_arg_val(5);
    constexpr uint32_t WPAGE = get_compile_time_arg_val(6);
    constexpr uint32_t DBG = get_compile_time_arg_val(7);  // debug: bit 0 skip X reads, bit 1 skip W reads
    constexpr auto aX = TensorAccessorArgs<8>();
    constexpr auto aW = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(1));
    const uint32_t nu = get_arg_val<uint32_t>(2);
    const uint32_t Rt = get_arg_val<uint32_t>(3 + 4 * get_compile_time_arg_val(3));  // row tiles of X
    constexpr uint32_t XB = 2048;
    Noc noc;
    CircularBuffer cx(0), cw(1);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t e = get_arg_val<uint32_t>(3 + 4 * u);
        const uint32_t r0 = get_arg_val<uint32_t>(4 + 4 * u);
        const uint32_t nr = get_arg_val<uint32_t>(5 + 4 * u);
        const uint32_t n0 = get_arg_val<uint32_t>(6 + 4 * u);
        cw.reserve_back(Kt * NN);
        if constexpr (DBG & 2) {
        } else if constexpr (WPAGE) {  // W pre-arranged: one page per (expert, column slice) holding its [Kt][NN] tiles
            noc.async_read(sW, cw, Kt * NN * WB, {.page_id = e * (Nt / NN) + n0 / NN}, {.offset_bytes = 0});
        } else {
            uint32_t off = 0;
            for (uint32_t k = 0; k < Kt; ++k) {
                const uint32_t base = (e * Kt + k) * Nt + n0;
                for (uint32_t n = 0; n < NN; ++n) {
                    noc.async_read(sW, cw, WB, {.page_id = base + n}, {.offset_bytes = off});
                    off += WB;
                }
            }
        }
        noc.async_read_barrier();
        cw.push_back(Kt * NN);
        for (uint32_t g = 0; g < nr; g += RT) {
            cx.reserve_back(RT * Kt);  // always RT row tiles (rows past the tensor: its last row tile)
            for (uint32_t r = 0; r < ((DBG & 1) ? 0 : RT); ++r) {
                const uint32_t row = r0 + g + r < Rt ? r0 + g + r : Rt - 1;
                for (uint32_t k = 0; k < Kt; ++k) {
                    noc.async_read(sX, cx, XB, {.page_id = row * Kt + k}, {.offset_bytes = (r * Kt + k) * XB});
                }
            }
            noc.async_read_barrier();
            cx.push_back(RT * Kt);
        }
    }
}
