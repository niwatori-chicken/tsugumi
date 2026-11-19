// GDN recurrence (fused, multi-token) — writer: final state (16 tiles, in place) and outputs O (4 tiles).
// CT args: 0 HPC, then TensorAccessorArgs for S, O. RT args: 0 S addr, 1 O addr, 2 first head.

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

void kernel_main() {
    constexpr uint32_t HPC = get_compile_time_arg_val(0);
    constexpr auto aS = TensorAccessorArgs<1>();
    constexpr auto aO = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(0));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(1));
    const uint32_t h0 = get_arg_val<uint32_t>(2);
    Noc noc;
    for (uint32_t hh = 0; hh < HPC; ++hh) {
        const uint32_t h = h0 + hh;
        write_tiles(noc, sS, c_Sout, h * 16, 16);
        write_tiles(noc, sO, c_Oout, h * 4, 4);
    }
    noc.async_write_barrier();
}
