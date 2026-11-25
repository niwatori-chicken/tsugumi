// Inkling prefill MoE for long chunks — writer (see ink_moex_reader.cpp). Per batch: once the compute has made the
// batch's shared act tiles (CB 13), arrive at core 0, wait for the release, free the previous batch's slots; then
// per entry of the batch and per row tile m it touches, a block of CTM placement tiles (CB 3: tile mi,
// [t % 32][r % 32] = 1 for token t = L[r] in row tile m, r in compact tile mi). The ring of blocks is zeroed once;
// each block's ones are cleared when its slot comes round again. Then the 2 MT output tiles (CB 16) -> OUT pages
// m * 72 + 2c + j; a final round keeps CB 13 alive until every core has read it.
// CB 17: lists from the reader (word 0 = U, word 1 + j = off | n << 16, tokens u16 at byte 1024).
// CT: 0 PS blocks in the ring, accessor OUT.  RT: 0 out, 1 c, 2 core 0 x, 3 core 0 y, 4 NB, 5 MT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t NPS = get_compile_time_arg_val(0);

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<1>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1), cx = get_arg_val<uint32_t>(2), cy = get_arg_val<uint32_t>(3);
    const uint32_t NB = get_arg_val<uint32_t>(4), MT = get_arg_val<uint32_t>(5), dbg = get_arg_val<uint32_t>(6);
    constexpr uint32_t TB = 2048;
    Noc noc;
    const uint32_t CTM = MT < 2 ? MT : 2;
    CircularBuffer cps(3), csh(13), co(16), cls(17);
    {
        cps.reserve_back(NPS * CTM);
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t a = cps.get_write_ptr();
        for (uint32_t o = 0; o < NPS * CTM * TB; o += MEM_ZEROS_SIZE) noc_async_read(z, a + o, MEM_ZEROS_SIZE);
        noc_async_read_barrier();
    }
    CircularBuffer crec(20);  // ones set in each ring slot (L1: the RISC-V stack is small)
    crec.reserve_back(1);
    volatile tt_l1_ptr uint16_t* rec = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(crec.get_write_ptr());
    volatile tt_l1_ptr uint8_t* recn = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(rec + NPS * 32);
    for (uint32_t s = 0; s < NPS; ++s) recn[s] = 0;
    uint32_t blk = 0;
    cls.wait_front(1);
    const uint32_t lb = cls.get_read_ptr();
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lb);
    volatile tt_l1_ptr uint16_t* tok = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lb + 1024);
    const uint32_t U = table[0];
    const uint32_t nbatch = (U + NB - 1) / NB;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint64_t arrive = get_noc_addr(cx, cy, get_semaphore(0));
    uint32_t prev = 0;  // tiles of the previous batch still in SHARE
    for (uint32_t b = 0; b < nbatch; ++b) {
        const uint32_t j0 = b * NB, n = (j0 + NB < U) ? NB : U - j0;
        csh.wait_front(prev + n * CTM);
        noc_semaphore_inc(arrive, 1);
        noc_semaphore_wait_min(go, b + 2);  // round 1: the readers' X shares
        if (prev) csh.pop_front(prev);
        prev = n * CTM;
        for (uint32_t j = j0; j < j0 + n; ++j) {
            const uint32_t o = table[1 + j] & 0xFFFF, cnt = table[1 + j] >> 16;
            volatile tt_l1_ptr uint16_t* L = tok + o;
            uint32_t r = 0;
            while (r < cnt) {
                const uint32_t m = L[r] >> 5;
                cps.reserve_back(CTM);
                volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cps.get_write_ptr());
                const uint32_t s = blk % NPS;
                for (uint32_t i = 0; i < recn[s] && !(dbg & 128); ++i) p[rec[s * 32 + i]] = 0;
                uint32_t k = 0;
                for (; r < cnt && (L[r] >> 5) == m; ++r) {
                    if (dbg & 128) continue;
                    const uint32_t pos = (r >> 5) * 1024 + elem_off(L[r] & 31, r & 31);
                    p[pos] = 0x3F80;  // 1.0
                    rec[s * 32 + k++] = pos;
                }
                recn[s] = k;
                cps.push_back(CTM);
                ++blk;
            }
        }
    }
    for (uint32_t i = 0; i < 2 * MT; ++i) {
        co.wait_front(1);
        noc.async_write(co, sO, TB, {.offset_bytes = 0}, {.page_id = (i >> 1) * 72 + 2 * c + (i & 1)});
        noc.async_write_barrier();
        co.pop_front(1);
    }
    noc_semaphore_inc(arrive, 1);
    noc_semaphore_wait_min(go, nbatch + 2);
    if (prev) csh.pop_front(prev);
    cls.pop_front(1);
    crec.push_back(1);
    noc_async_atomic_barrier();
}
