// GQA split-K attention — combine reader. Core (g, r, dg): row tile r of kv group g, value tiles
// [dg*DT, dg*DT+DT). Only the non-empty splits s < NA = min(S, ceil((POS[0]+T)/C)) are read.
// Pushes info (NA), m_s, l_s (s < NA), then O_s (DT tiles) per split.  PART layout: see gqa_attn_writer.cpp.
// CT: 0 RT, 1 HDT, 2 NKV, 3 S, 4 DT, 5 T, 6 C, 7 POSB, then accessors PART, POS.  RT: 0, 1 addrs (POS: this core's lane), 2 g, 3 r, 4 dg, 5 first split of the lane in PART
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_MS = 0, c_LS = 1, c_OS = 2, c_INFO = 9;

void kernel_main() {
    constexpr uint32_t RT = get_compile_time_arg_val(0);
    constexpr uint32_t HDT = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t DT = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t C = get_compile_time_arg_val(6);
    constexpr uint32_t POSB = get_compile_time_arg_val(7);
    constexpr auto aP = TensorAccessorArgs<8>();
    constexpr auto aX = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(0));
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(1));
    const uint32_t g = get_arg_val<uint32_t>(2);
    const uint32_t r = get_arg_val<uint32_t>(3);
    const uint32_t dg = get_arg_val<uint32_t>(4);
    const uint32_t soff = get_arg_val<uint32_t>(5);
    constexpr uint32_t FB = 4096;
    constexpr uint32_t J = HDT + 2;
    Noc noc;
    CircularBuffer info(c_INFO), ms(c_MS), ls(c_LS), os(c_OS);
    info.reserve_back(1);
    noc.async_read(sX, info, POSB, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(info.get_write_ptr());
    const uint32_t nch = (w[0] + T + C - 1) / C;
    const uint32_t NA = nch < S ? nch : S;
    w[0] = NA;
    info.push_back(1);
    ms.reserve_back(S);
    ls.reserve_back(S);
    for (uint32_t s = 0; s < NA; ++s) {
        const uint32_t page0 = (((soff + s) * NKV + g) * RT + r) * J;
        noc.async_read(sP, ms, FB, {.page_id = page0 + HDT}, {.offset_bytes = s * FB});
        noc.async_read(sP, ls, FB, {.page_id = page0 + HDT + 1}, {.offset_bytes = s * FB});
    }
    noc.async_read_barrier();
    ms.push_back(S);
    ls.push_back(S);
    for (uint32_t s = 0; s < NA; ++s) {
        const uint32_t page0 = (((soff + s) * NKV + g) * RT + r) * J + dg * DT;
        os.reserve_back(DT);
        for (uint32_t d = 0; d < DT; ++d) {
            noc.async_read(sP, os, FB, {.page_id = page0 + d}, {.offset_bytes = d * FB});
        }
        noc.async_read_barrier();
        os.push_back(DT);
    }
}
