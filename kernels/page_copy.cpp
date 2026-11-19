// Copy a page range between two interleaved DRAM tensors with the same page size:
//   dst[DOFF + i] = src[SOFF + i],  i in [lo, hi)  (this core's share)
// SOFF / DOFF come from a uint32 row-major tensor OFF ([soff, doff]) so the op can be traced
// with offsets set at run time. CT: 0 PB (page bytes), 1 NBUF, then accessors SRC, DST, OFF.  RT: 0..2 addrs, 3 lo, 4 hi
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t PB = get_compile_time_arg_val(0);
    constexpr uint32_t NBUF = get_compile_time_arg_val(1);
    constexpr auto aS = TensorAccessorArgs<2>();
    constexpr auto aD = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    constexpr auto aO = TensorAccessorArgs<aD.next_compile_time_args_offset()>();
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(0));
    const auto sD = TensorAccessor(aD, get_arg_val<uint32_t>(1));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(2));
    const uint32_t lo = get_arg_val<uint32_t>(3);
    const uint32_t hi = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer buf(0), ob(1);
    ob.reserve_back(1);
    noc.async_read(sO, ob, 32, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob.get_write_ptr());
    const uint32_t soff = o[0], doff = o[1];
    buf.reserve_back(NBUF);  // used as scratch: never pushed, so read ptr == write ptr
    for (uint32_t i = lo; i < hi; i += NBUF) {
        const uint32_t n = (hi - i < NBUF) ? hi - i : NBUF;
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_read(sS, buf, PB, {.page_id = soff + i + j}, {.offset_bytes = j * PB});
        }
        noc.async_read_barrier();
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_write(buf, sD, PB, {.offset_bytes = j * PB}, {.page_id = doff + i + j});
        }
        noc.async_write_barrier();
    }
}
