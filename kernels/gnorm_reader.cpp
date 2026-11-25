// Gated RMSNorm on the head-major GDN output (large-T prefill) — reader.
// Unit u = (row tile tt = u / NV, head h = u % NV):
//   O : [NV, T, 128] TILE (chunk_gated_delta_rule output_head_major), tiles (h*NT + tt)*4 + j
//   Y : [1,1,T,YW*32] bf16 TILE (in-projection output), z of head h = column tiles ZOFF + h*4 + j
//   C : [1,1,6*32,32] fp32 TILE constants: 0 ones/128, 1 eps, 2..5 norm weight (rows replicated)
// CT: 0 NT, 1 NV, 2 YW, 3 ZOFF, 4 OTB (O tile bytes), then accessors O, Y, C.  RT: 0..2 addrs, 3 u0, 4 u1
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_O = 0, c_Z = 1, c_C = 2;

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t NV = get_compile_time_arg_val(1);
    constexpr uint32_t YW = get_compile_time_arg_val(2);
    constexpr uint32_t ZOFF = get_compile_time_arg_val(3);
    constexpr uint32_t OTB = get_compile_time_arg_val(4);
    constexpr auto aO = TensorAccessorArgs<5>();
    constexpr auto aY = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aY.next_compile_time_args_offset()>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const auto sY = TensorAccessor(aY, get_arg_val<uint32_t>(1));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(2));
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t u1 = get_arg_val<uint32_t>(4);
    constexpr uint32_t TB = 2048, FB = 4096;
    Noc noc;
    CircularBuffer co(c_O), cz(c_Z), cc(c_C);
    cc.reserve_back(6);
    for (uint32_t k = 0; k < 6; ++k) {
        noc.async_read(sC, cc, FB, {.page_id = k}, {.offset_bytes = k * FB});
    }
    noc.async_read_barrier();
    cc.push_back(6);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t tt = u / NV, h = u % NV;
        co.reserve_back(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_read(sO, co, OTB, {.page_id = (h * NT + tt) * 4 + j}, {.offset_bytes = j * OTB});
        }
        cz.reserve_back(4);
        for (uint32_t j = 0; j < 4; ++j) {
            noc.async_read(sY, cz, TB, {.page_id = tt * YW + ZOFF + h * 4 + j}, {.offset_bytes = j * TB});
        }
        noc.async_read_barrier();
        co.push_back(4);
        cz.push_back(4);
    }
}
