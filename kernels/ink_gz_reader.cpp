// Inkling message gate, phase A — reader. Core c takes KT column tiles of the vector [h | m] (h for c < NC/2, m
// after): the source tiles (row 0 = the token) and the matching KT x RT tiles of D ([2H, R] = down_weight^T).
// CB 0: V (KT), 1: D (KT * RT), 2: SC (reduce scaler).  CT: accessors H, D (DRAM), SC (DRAM), M
// RT: 0 src, 1 d, 2 sc, 3 first source tile, 4 first D row tile, 5 KT, 6 RT, 7 source is M
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aS = TensorAccessorArgs<0>();  // h
    constexpr auto aD = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aD.next_compile_time_args_offset()>();
    constexpr auto aM = TensorAccessorArgs<aC.next_compile_time_args_offset()>();  // m (may be in another memory)
    const auto sD = TensorAccessor(aD, get_arg_val<uint32_t>(1));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(2));
    const uint32_t t0 = get_arg_val<uint32_t>(3), k0 = get_arg_val<uint32_t>(4);
    const uint32_t KT = get_arg_val<uint32_t>(5), RT = get_arg_val<uint32_t>(6);
    Noc noc;
    CircularBuffer cv(0), cd(1), cs(2);
    cs.reserve_back(1);
    noc.async_read(sC, cs, 2048, {.page_id = 0}, {.offset_bytes = 0});
    cv.reserve_back(KT);
    if (get_arg_val<uint32_t>(7)) {
        const auto sM = TensorAccessor(aM, get_arg_val<uint32_t>(0));
        for (uint32_t j = 0; j < KT; ++j) noc.async_read(sM, cv, 2048, {.page_id = t0 + j}, {.offset_bytes = j * 2048});
    } else {
        const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(0));
        for (uint32_t j = 0; j < KT; ++j) noc.async_read(sS, cv, 2048, {.page_id = t0 + j}, {.offset_bytes = j * 2048});
    }
    cd.reserve_back(KT * RT);
    for (uint32_t j = 0; j < KT; ++j)
        for (uint32_t r = 0; r < RT; ++r)
            noc.async_read(sD, cd, 2048, {.page_id = (k0 + j) * RT + r}, {.offset_bytes = (j * RT + r) * 2048});
    noc.async_read_barrier();
    cs.push_back(1);
    cv.push_back(KT);
    cd.push_back(KT * RT);
}
