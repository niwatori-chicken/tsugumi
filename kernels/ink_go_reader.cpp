// Inkling message gate, phase B — reader. Core takes CPT output column tiles j0..: the phase-A partials (NC x
// (RT + 1) float32 tiles), FG ([R, 2H]: forget | gain) tiles of its columns, h and m tiles of its columns.
// CB 0: PART, 1: FG (f tiles RT x CPT, then g tiles), 2: H (CPT), 3: M (CPT).
// CT: accessors PART (DRAM f32), FG (DRAM), SRC (L1).  RT: 0 part, 1 fg, 2 h, 3 m, 4 NC, 5 RT, 6 j0, 7 CPT, 8 NCOL (FG column tiles), 9 HT (column tiles of h)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aP = TensorAccessorArgs<0>();
    constexpr auto aF = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    constexpr auto aS = TensorAccessorArgs<aF.next_compile_time_args_offset()>();
    const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(0));
    const auto sF = TensorAccessor(aF, get_arg_val<uint32_t>(1));
    const auto sH = TensorAccessor(aS, get_arg_val<uint32_t>(2));
    const auto sM = TensorAccessor(aS, get_arg_val<uint32_t>(3));
    const uint32_t NC = get_arg_val<uint32_t>(4), RT = get_arg_val<uint32_t>(5), j0 = get_arg_val<uint32_t>(6);
    const uint32_t CPT = get_arg_val<uint32_t>(7), NCOL = get_arg_val<uint32_t>(8), HT = get_arg_val<uint32_t>(9);
    Noc noc;
    CircularBuffer cp(0), cf(1), ch(2), cm(3);
    cp.reserve_back(NC * (RT + 1));
    for (uint32_t i = 0; i < NC * (RT + 1); ++i) noc.async_read(sP, cp, 4096, {.page_id = i}, {.offset_bytes = i * 4096});
    cf.reserve_back(2 * RT * CPT);
    for (uint32_t g = 0; g < 2; ++g)
        for (uint32_t k = 0; k < RT; ++k)
            for (uint32_t j = 0; j < CPT; ++j)
                noc.async_read(sF, cf, 2048, {.page_id = k * NCOL + g * HT + j0 + j},
                               {.offset_bytes = ((g * RT + k) * CPT + j) * 2048});
    ch.reserve_back(CPT);
    cm.reserve_back(CPT);
    for (uint32_t j = 0; j < CPT; ++j) {
        noc.async_read(sH, ch, 2048, {.page_id = j0 + j}, {.offset_bytes = j * 2048});
        noc.async_read(sM, cm, 2048, {.page_id = j0 + j}, {.offset_bytes = j * 2048});
    }
    noc.async_read_barrier();
    cp.push_back(NC * (RT + 1));
    cf.push_back(2 * RT * CPT);
    ch.push_back(CPT);
    cm.push_back(CPT);
}
