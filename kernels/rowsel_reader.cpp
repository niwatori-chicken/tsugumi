// Row gather (conv history of the large-T prefill) — reader. Output column tile c = MA @ XA[ra, c] + MB @ XB[rb, c]
// (MA, MB: 32x32 0/1 row-selection tiles). XA, XB [1,1,R,W*32] bf16 TILE (row strides AW, BW tiles),
// M [1,1,64,32] bf16 TILE (MA; MB).
// CT: 0 AW, 1 BW, then accessors XA, XB, M.  RT: 0..2 addrs, 3 ra, 4 rb, 5 c0, 6 c1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t AW = get_compile_time_arg_val(0);
    constexpr uint32_t BW = get_compile_time_arg_val(1);
    constexpr auto aA = TensorAccessorArgs<2>();
    constexpr auto aB = TensorAccessorArgs<aA.next_compile_time_args_offset()>();
    constexpr auto aM = TensorAccessorArgs<aB.next_compile_time_args_offset()>();
    const auto sA = TensorAccessor(aA, get_arg_val<uint32_t>(0));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(1));
    const auto sM = TensorAccessor(aM, get_arg_val<uint32_t>(2));
    const uint32_t ra = get_arg_val<uint32_t>(3);
    const uint32_t rb = get_arg_val<uint32_t>(4);
    const uint32_t c0 = get_arg_val<uint32_t>(5);
    const uint32_t c1 = get_arg_val<uint32_t>(6);
    constexpr uint32_t TB = 2048;
    Noc noc;
    CircularBuffer cx(0), cm(1);
    cm.reserve_back(2);
    noc.async_read(sM, cm, TB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(sM, cm, TB, {.page_id = 1}, {.offset_bytes = TB});
    noc.async_read_barrier();
    cm.push_back(2);
    for (uint32_t c = c0; c < c1; ++c) {
        cx.reserve_back(2);
        noc.async_read(sA, cx, TB, {.page_id = ra * AW + c}, {.offset_bytes = 0});
        noc.async_read(sB, cx, TB, {.page_id = rb * BW + c}, {.offset_bytes = TB});
        noc.async_read_barrier();
        cx.push_back(2);
    }
}
