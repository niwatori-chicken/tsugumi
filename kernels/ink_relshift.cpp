// Inkling prefill: the relative bias by distance -> by cache slot, a per-row circular shift (replaces ttnn.gather
// with a [heads, rows, slots] index tensor). With rel[.., d] the bias at distance d (columns >= L: 0):
// out[h, t, s] = rel[h, t, (base + t - s) mod M] for s < M, rel[h, t, L - 1] beyond (must be 0: the bias at
// distances >= the window). The input holds the
// columns reversed (in[.., y] = rel[.., L - 1 - y]): the slots of a tile row then read increasing columns, copied as
// 32-bit pairs. in [1, NH, T_in, L] bf16 tiles, out [1, NH, TO, OW] bf16 tiles, t = the
// global row (the output covers rows m0 * 32 .. m0 * 32 + TO); base = BASE[k] (int32, row-major, per chunk).
// Runs as reader and writer kernels (both RISC-Vs), each on its own CBs and units (h, output row tile).
// CT: 0 LT (L / 32), 1 OT (OW / 32), 2 first CB id (CBs id, id + 1, id + 4), accessors IN, OUT, BASE.
// RT: 0 in, 1 out, 2 base, 3 k, 4 m0, 5 MT_in (T_in / 32), 6 MTo, 7 first unit, 8 unit stride, 9 units, 10 M
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t LT = get_compile_time_arg_val(0), OT = get_compile_time_arg_val(1), CB0 = get_compile_time_arg_val(2);

void kernel_main() {
    constexpr auto aI = TensorAccessorArgs<3>();
    constexpr auto aO = TensorAccessorArgs<aI.next_compile_time_args_offset()>();
    constexpr auto aB = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(0));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(1));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(2));
    const uint32_t k = get_arg_val<uint32_t>(3), m0 = get_arg_val<uint32_t>(4), MTi = get_arg_val<uint32_t>(5);
    const uint32_t MTo = get_arg_val<uint32_t>(6), u0 = get_arg_val<uint32_t>(7), us = get_arg_val<uint32_t>(8);
    const uint32_t nu = get_arg_val<uint32_t>(9);
    const uint32_t M = get_arg_val<uint32_t>(10);
    const uint32_t dbg = get_arg_val<uint32_t>(11);  // 1: no element loop, 2: no reads, 4: no writes  // the modulus (ring slots R or window W; >= L: columns >= L are 0)
    constexpr uint32_t TB = 2048, L = LT * 32;
    CircularBuffer ci(CB0), co(CB0 + 1);
    ci.reserve_back(LT);
    const uint32_t in = ci.get_write_ptr();
    // base (the first 8 int32 of BASE) via the input CB's start, then the row tiles overwrite it
    noc_async_read(sB.get_noc_addr(0), in, 32);
    noc_async_read_barrier();
    const uint32_t base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in)[k] % M;
    const uint16_t* src = reinterpret_cast<const uint16_t*>(in);
    const uint32_t* srcw = reinterpret_cast<const uint32_t*>(in);
    co.reserve_back(2);
    const uint32_t obase = co.get_write_ptr();
    for (uint32_t u = u0; u < nu; u += us) {
        const uint32_t h = u / MTo, mo = u % MTo, m = m0 + mo;
        noc_async_write_barrier();  // (the previous unit's last tile is written from obase, not from in)
        for (uint32_t j = 0; j < LT && !(dbg & 2); ++j) noc_async_read(sI.get_noc_addr((h * MTi + m) * LT + j), in + j * TB, TB);
        noc_async_read_barrier();
        for (uint32_t jt = 0; jt < OT; ++jt) {
            const uint32_t ob = obase + (jt & 1) * TB;  // two output buffers: compute one while the other is written
            uint16_t* dst = reinterpret_cast<uint16_t*>(ob);
            uint32_t* dstw = reinterpret_cast<uint32_t*>(ob);
            const bool in_m = 32 * jt + 31 < M;  // every slot of the tile below the modulus
            for (uint32_t r = 0; r < 32 && !(dbg & 1); ++r) {
                const uint32_t rp = (r >> 4) * 512 + (r & 15) * 16, rpw = rp >> 1;  // row part (elements / words)
                int32_t col = (int32_t)((base + 32 * m + r) % M) - (int32_t)(32 * jt);  // column of slot 32 jt
                while (col < 0) col += M;
                if (in_m && col >= 31 && col < (int32_t)L) {  // one run: reversed columns y0 .. y0 + 31
                    const uint32_t y0 = L - 1 - col;
                    auto W = [&](uint32_t y) { return rpw + (y >> 5) * 512 + ((y >> 4) & 1) * 128 + ((y & 15) >> 1); };
                    if (!(y0 & 1)) {
                        for (uint32_t i = 0; i < 16; ++i) dstw[rpw + (i >> 3) * 128 + (i & 7)] = srcw[W(y0 + 2 * i)];
                    } else {
                        uint32_t prev = srcw[W(y0 - 1)];
                        for (uint32_t i = 0; i < 16; ++i) {
                            const uint32_t nxt = srcw[W(y0 + 1 + 2 * i)];
                            dstw[rpw + (i >> 3) * 128 + (i & 7)] = (prev >> 16) | (nxt << 16);
                            prev = nxt;
                        }
                    }
                    continue;
                }
                if (32 * jt >= M || (in_m && col >= (int32_t)(L + 31))) {  // every slot of the row at distance >= L: 0
                    for (uint32_t i = 0; i < 16; ++i) dstw[rpw + (i >> 3) * 128 + (i & 7)] = 0;
                    continue;
                }
                for (uint32_t f = 0; f < 2; ++f) {  // element by element (wrap, columns >= L, slots >= M)
                    uint16_t* d = dst + rp + f * 256;
                    for (uint32_t c = 0; c < 16; ++c) {
                        const uint32_t s_ = 32 * jt + 16 * f + c;
                        const uint32_t sc = (s_ < M && (uint32_t)col < L) ? (uint32_t)col : L - 1;
                        const uint32_t y = L - 1 - sc;
                        d[c] = src[rp + (y >> 5) * 1024 + ((y & 31) >> 4) * 256 + (y & 15)];
                        col = col == 0 ? (int32_t)M - 1 : col - 1;
                    }
                }
            }
            noc_async_write_barrier();
            if (!(dbg & 4)) noc_async_write(ob, sO.get_noc_addr((h * MTo + mo) * OT + jt), TB);
        }
    }
    noc_async_write_barrier();
}
