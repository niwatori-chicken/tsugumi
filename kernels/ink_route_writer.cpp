// Inkling decode MoE routing — writer: the K experts with the largest s + bias (ties: lowest id), weights
// s_i / sum(s) * KSCALE. Writes IDS (uint16 row-major [1,1,1,K], L1) and WCOL ([1,K,1,1] bf16 tiles: value at [0, 0]
// of tile i; L1).  CT: accessors IDS, WCOL.  RT: 0 ids, 1 wcol, 2 tiles, 3 K, 4 KSCALE (float bits)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr auto aI = TensorAccessorArgs<0>();
    constexpr auto aW = TensorAccessorArgs<aI.next_compile_time_args_offset()>();
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(0));
    const auto sW = TensorAccessor(aW, get_arg_val<uint32_t>(1));
    const uint32_t nt = get_arg_val<uint32_t>(2), K = get_arg_val<uint32_t>(3);
    const uint32_t kbits = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer cs(2), csb(3), cscr(4);
    cs.wait_front(nt);
    csb.wait_front(nt);
    volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cs.get_read_ptr());
    volatile tt_l1_ptr uint32_t* SB = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csb.get_read_ptr());
    const uint32_t E = nt * 32;
    uint32_t key[256];
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t b = SB[(e >> 5) * 1024 + elem_off(0, e & 31)];
        key[e] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);  // order-preserving
    }
    uint32_t sel[16];
    float sv[16];
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t best = 0, bk = 0;
        bool found = false;
        for (uint32_t e = 0; e < E; ++e) {
            if (!found || key[e] > bk) {
                bk = key[e];
                best = e;
                found = true;
            }
        }
        key[best] = 0;  // below every real key
        sel[k] = best;
        uint32_t sbits = S[(best >> 5) * 1024 + elem_off(0, best & 31)];
        float f;
        __builtin_memcpy(&f, &sbits, 4);
        sv[k] = f;
        sum += f;
    }
    float ks;
    __builtin_memcpy(&ks, &kbits, 4);
    const float norm = ks / sum;
    cscr.reserve_back(1);
    const uint32_t base = cscr.get_write_ptr();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    for (uint32_t k = 0; k < K; ++k) ids[k] = (uint16_t)sel[k];
    noc.async_write(cscr, sI, K * 2, {.offset_bytes = 0}, {.page_id = 0});
    for (uint32_t k = 0; k < K; ++k) {
        volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 64 + k * 32);
        float w = sv[k] * norm;
        uint32_t wb;
        __builtin_memcpy(&wb, &w, 4);
        wb += 0x7FFFu + ((wb >> 16) & 1u);  // round to nearest even
        row[0] = (uint16_t)(wb >> 16);
        for (uint32_t i = 1; i < 16; ++i) row[i] = 0;
        noc.async_write(cscr, sW, 32, {.offset_bytes = 64 + k * 32}, {.page_id = k});
    }
    noc.async_write_barrier();
    cs.pop_front(nt);
    csb.pop_front(nt);
}
