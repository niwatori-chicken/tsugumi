// GQA split-K attention — writer: partial O (RT x HDT), m (RT), l (RT) fp32 tiles to PART.
// PART page of (s, g, r, j) = ((s * NKV + g) * RT + r) * (HDT + 2) + j   (j < HDT: O, HDT: m, HDT+1: l)
// CT: 0 RT, 1 HDT, 2 NKV, then accessor PART.  RT: 0 addr, 1 g, 2 s
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_FO = 16, c_FM = 17, c_FL = 18;

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t HDT = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr auto aO = TensorAccessorArgs<3>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t s = get_arg_val<uint32_t>(2);
    constexpr uint32_t FB = 4096;
    constexpr uint32_t J = HDT + 2;
    Noc noc;
    CircularBuffer fo(c_FO), fm(c_FM), fl(c_FL);
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t page0 = ((s * NKV + g) * RT + r) * J;
        for (uint32_t d = 0; d < HDT; ++d) {
            fo.wait_front(1);
            noc.async_write(fo, sO, FB, {.offset_bytes = 0}, {.page_id = page0 + d});
            noc.async_write_barrier();
            fo.pop_front(1);
        }
    }
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t page0 = ((s * NKV + g) * RT + r) * J;
        fm.wait_front(1);
        noc.async_write(fm, sO, FB, {.offset_bytes = 0}, {.page_id = page0 + HDT});
        fl.wait_front(1);
        noc.async_write(fl, sO, FB, {.offset_bytes = 0}, {.page_id = page0 + HDT + 1});
        noc.async_write_barrier();
        fm.pop_front(1);
        fl.pop_front(1);
    }
}
