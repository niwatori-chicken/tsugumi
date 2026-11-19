// GDN recurrence (fused, multi-token) — reader.
// For each head this core owns, read Q/K/V (rows = tokens), K^T, beta tile, T decay tiles and the
// 16-tile fp32 state into CBs. The T row-selector tiles E are read once per core.
//
// CT args: 0 T, 1 heads_per_core, then TensorAccessorArgs for Q, K, KT, V, B, G, E, S (8 tensors).
// RT args: 0..7 buffer addresses (Q, K, KT, V, B, G, E, S), 8 first head index.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_S0 = 0, c_Q = 2, c_K = 3, c_KT = 4, c_V = 5, c_B = 6, c_G = 7, c_E = 8;

template <typename Acc>
FORCE_INLINE void read_tiles(Noc& noc, const Acc& acc, uint32_t cb_id, uint32_t first, uint32_t n) {
    CircularBuffer cb(cb_id);
    const uint32_t tsz = get_tile_size(cb_id);
    cb.reserve_back(n);
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, cb, tsz, {.page_id = first + i}, {.offset_bytes = i * tsz});
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);
    constexpr auto aQ = TensorAccessorArgs<2>();
    constexpr auto aK = TensorAccessorArgs<aQ.next_compile_time_args_offset()>();
    constexpr auto aKT = TensorAccessorArgs<aK.next_compile_time_args_offset()>();
    constexpr auto aV = TensorAccessorArgs<aKT.next_compile_time_args_offset()>();
    constexpr auto aB = TensorAccessorArgs<aV.next_compile_time_args_offset()>();
    constexpr auto aG = TensorAccessorArgs<aB.next_compile_time_args_offset()>();
    constexpr auto aE = TensorAccessorArgs<aG.next_compile_time_args_offset()>();
    constexpr auto aS = TensorAccessorArgs<aE.next_compile_time_args_offset()>();

    const auto sQ = TensorAccessor(aQ, get_arg_val<uint32_t>(0));
    const auto sK = TensorAccessor(aK, get_arg_val<uint32_t>(1));
    const auto sKT = TensorAccessor(aKT, get_arg_val<uint32_t>(2));
    const auto sV = TensorAccessor(aV, get_arg_val<uint32_t>(3));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(4));
    const auto sG = TensorAccessor(aG, get_arg_val<uint32_t>(5));
    const auto sE = TensorAccessor(aE, get_arg_val<uint32_t>(6));
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(7));
    const uint32_t h0 = get_arg_val<uint32_t>(8);

    Noc noc;
    read_tiles(noc, sE, c_E, 0, T);
    for (uint32_t hh = 0; hh < HPC; ++hh) {
        const uint32_t h = h0 + hh;
        read_tiles(noc, sS, c_S0, h * 16, 16);
        read_tiles(noc, sG, c_G, h * T, T);
        read_tiles(noc, sK, c_K, h * 4, 4);
        read_tiles(noc, sV, c_V, h * 4, 4);
        read_tiles(noc, sB, c_B, h, 1);
        read_tiles(noc, sKT, c_KT, h * 4, 4);
        read_tiles(noc, sQ, c_Q, h * 4, 4);
    }
}
