// Fused GDN — writer: state snapshot after every token (slot r0 + t), then the output tiles.
// CT args: 0 HPC, 1 NSLOT (rows of this core, or 1 = final state only), 2 NV, 3 PART (several cores per head: write
//          only rows [r0, r1) of the output tiles), then TensorAccessorArgs for S, O.
// RT args: 0 S addr, 1 O addr, 2 first head, 3 r0 (first row / snapshot slot of this core), 4 r1 (end of its output rows)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_Sout = 16, c_Oout = 17;

template <typename Acc>
FORCE_INLINE void write_tiles(Noc& noc, const Acc& acc, uint32_t cb_id, uint32_t first, uint32_t n) {
    CircularBuffer cb(cb_id);
    const uint32_t tsz = get_tile_size(cb_id);
    for (uint32_t i = 0; i < n; ++i) {
        cb.wait_front(1);
        noc.async_write(cb, acc, tsz, {.offset_bytes = 0}, {.page_id = first + i});
        noc.async_writes_flushed();
        cb.pop_front(1);
    }
}

// rows [r0, r1) of n bf16 tiles: per 16-row block, the two faces' runs of (rows x 32 bytes)
template <typename Acc>
FORCE_INLINE void write_rows(Noc& noc, const Acc& acc, uint32_t cb_id, uint32_t first, uint32_t n, uint32_t r0,
                             uint32_t r1) {
    CircularBuffer cb(cb_id);
    for (uint32_t i = 0; i < n; ++i) {
        cb.wait_front(1);
        for (uint32_t b = 0; b < 2; ++b) {
            const uint32_t lo = r0 > b * 16 ? r0 : b * 16;
            const uint32_t hi = r1 < b * 16 + 16 ? r1 : b * 16 + 16;
            if (lo >= hi) {
                continue;
            }
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t off = (((b * 2 + f) * 256) + (lo & 15) * 16) * 2;
                noc.async_write(cb, acc, (hi - lo) * 32, {.offset_bytes = off},
                                {.page_id = first + i, .offset_bytes = off});
            }
        }
        noc.async_writes_flushed();
        cb.pop_front(1);
    }
}

void kernel_main() {
    constexpr uint32_t HPC = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t PART = get_compile_time_arg_val(3);
    constexpr auto aS = TensorAccessorArgs<4>();
    constexpr auto aO = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(0));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(1));
    const uint32_t h0 = get_arg_val<uint32_t>(2);
    const uint32_t r0 = get_arg_val<uint32_t>(3);
    const uint32_t r1 = get_arg_val<uint32_t>(4);
    const uint32_t sb = T == 1 ? 0 : r0;  // final-state-only (prefill): slot 0
    Noc noc;
    for (uint32_t hh = 0; hh < HPC; ++hh) {
        const uint32_t h = h0 + hh;
        for (uint32_t t = 0; t < T; ++t) {
            write_tiles(noc, sS, c_Sout, ((sb + t) * NV + h) * 16, 16);
        }
        if constexpr (PART) {
            write_rows(noc, sO, c_Oout, h * 4, 4, r0, r1);
        } else {
            write_tiles(noc, sO, c_Oout, h * 4, 4);
        }
    }
    noc.async_write_barrier();
}
