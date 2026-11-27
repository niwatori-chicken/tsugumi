// Inkling prefill: the attention mask [1, NH, rows, OW] (units of 1 / scale) built by selection only (no float math):
// per head h, row t (global row; position P_t = p0 + t) and cache column s:
//   sliding (mode 0): slot s < SC: distance d = (c0 + t - s) mod M (M = R ring slots); valid iff d < W and d <= P_t:
//                     rel(d), else NEG;
//   full    (mode 1): position s < SC: d = P_t - s; d < 0: NEG; d < W: rel(d); else 0;
//   s == SC: the sink logit sink[h, t] (column 0 of its tile); s > SC: NEG.
// rel(d) = in[h, t, L - 1 - d]: the relative bias by distance with the columns REVERSED (runs of slots then read
// increasing columns, copied as 32-bit pairs).
// Runs as reader and writer kernels (both RISC-Vs), each on its own CBs and units (h, output row tile).
// CT: 0 LT (L / 32), 1 OT (OW / 32), 2 first CB id (id, id + 1), 3 mode, accessors IN, SINK, OUT, BASE.
// RT: 0 in, 1 sink, 2 out, 3 base, 4 m0, 5 MT_in, 6 MTo, 7 first unit, 8 unit stride, 9 units, 10 M, 11 W, 12 SC
// BASE (int32 row-major): [0] = p0 % R, [2] = p0.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t LT = get_compile_time_arg_val(0), OT = get_compile_time_arg_val(1), CB0 = get_compile_time_arg_val(2);
constexpr uint32_t MODE = get_compile_time_arg_val(3);
constexpr uint16_t NEG = 0xCE6E;  // bf16 -1e9

void kernel_main() {
    constexpr auto aI = TensorAccessorArgs<4>();
    constexpr auto aS = TensorAccessorArgs<aI.next_compile_time_args_offset()>();
    constexpr auto aO = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    constexpr auto aB = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(0));
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(1));
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(2));
    const auto sB = TensorAccessor(aB, get_arg_val<uint32_t>(3));
    const uint32_t m0 = get_arg_val<uint32_t>(4), MTi = get_arg_val<uint32_t>(5), MTo = get_arg_val<uint32_t>(6);
    const uint32_t u0 = get_arg_val<uint32_t>(7), us = get_arg_val<uint32_t>(8), nu = get_arg_val<uint32_t>(9);
    const uint32_t M = get_arg_val<uint32_t>(10), W = get_arg_val<uint32_t>(11), SC = get_arg_val<uint32_t>(12);
    constexpr uint32_t TB = 2048, L = LT * 32;
    CircularBuffer ci(CB0), co(CB0 + 1);
    ci.reserve_back(LT + 1);  // the input row tile, then the sink tile
    const uint32_t in = ci.get_write_ptr(), sk = in + LT * TB;
    noc_async_read(sB.get_noc_addr(0), in, 32);
    noc_async_read_barrier();
    const uint32_t c0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in)[0];
    const uint32_t p0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in)[2];
    const uint16_t* src = reinterpret_cast<const uint16_t*>(in);
    const uint32_t* srcw = reinterpret_cast<const uint32_t*>(in);
    const uint16_t* sink = reinterpret_cast<const uint16_t*>(sk);
    co.reserve_back(4);  // two output buffers, a zero tile, a NEG tile
    const uint32_t obase = co.get_write_ptr(), zt = obase + 2 * TB, nt = obase + 3 * TB;
    {
        uint32_t* z = reinterpret_cast<uint32_t*>(zt);
        uint32_t* q = reinterpret_cast<uint32_t*>(nt);
        for (uint32_t i = 0; i < 512; ++i) {
            z[i] = 0;
            q[i] = ((uint32_t)NEG << 16) | NEG;
        }
    }
    constexpr uint32_t NEGW = ((uint32_t)NEG << 16) | NEG;
    for (uint32_t u = u0; u < nu; u += us) {
        const uint32_t h = u / MTo, mo = u % MTo, m = m0 + mo;
        noc_async_write_barrier();
        for (uint32_t j = 0; j < LT; ++j) noc_async_read(sI.get_noc_addr((h * MTi + m) * LT + j), in + j * TB, TB);
        noc_async_read(sS.get_noc_addr(h * MTi + m), sk, TB);
        noc_async_read_barrier();
        uint32_t nb = 0;  // tiles built in the output buffers (alternating)
        for (uint32_t jt = 0; jt < OT; ++jt) {
            // whole tile constant (no sink column, every row all NEG / all 0): written from the constant tiles
            {
                const uint32_t s0 = 32 * jt;
                int32_t kind = -1;  // 0: zeros, 1: NEG
                if (s0 > SC) {
                    kind = 1;
                } else if (s0 + 31 < SC) {
                    if constexpr (MODE == 1) {
                        const int32_t dmax = (int32_t)(p0 + 32 * m + 31) - (int32_t)s0;   // last row, first slot
                        const int32_t dmin = (int32_t)(p0 + 32 * m) - (int32_t)(s0 + 31);  // first row, last slot
                        if (dmax < 0) kind = 1;                      // future for every row
                        else if (dmin >= (int32_t)W) kind = 0;       // beyond the window for every row
                    }
                }
                if (kind >= 0) {
                    noc_async_write(kind ? nt : zt, sO.get_noc_addr((h * MTo + mo) * OT + jt), TB);
                    continue;
                }
            }
            const uint32_t ob = obase + (nb++ & 1) * TB;
            uint16_t* dst = reinterpret_cast<uint16_t*>(ob);
            uint32_t* dstw = reinterpret_cast<uint32_t*>(ob);
            const uint32_t s0 = 32 * jt;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t rp = (r >> 4) * 512 + (r & 15) * 16, rpw = rp >> 1;
                const uint32_t t = 32 * m + r, P = p0 + t;
                auto fill = [&](uint32_t w) {
                    for (uint32_t i = 0; i < 16; ++i) dstw[rpw + (i >> 3) * 128 + (i & 7)] = w;
                };
                auto run = [&](uint32_t d0) {  // distances d0, d0 - 1, .., d0 - 31: reversed columns y0 ..
                    const uint32_t y0 = L - 1 - d0;
                    auto Wd = [&](uint32_t y) { return rpw + (y >> 5) * 512 + ((y >> 4) & 1) * 128 + ((y & 15) >> 1); };
                    if (!(y0 & 1)) {
                        for (uint32_t i = 0; i < 16; ++i) dstw[rpw + (i >> 3) * 128 + (i & 7)] = srcw[Wd(y0 + 2 * i)];
                    } else {
                        uint32_t prev = srcw[Wd(y0 - 1)];
                        for (uint32_t i = 0; i < 16; ++i) {
                            const uint32_t nxt = srcw[Wd(y0 + 1 + 2 * i)];
                            dstw[rpw + (i >> 3) * 128 + (i & 7)] = (prev >> 16) | (nxt << 16);
                            prev = nxt;
                        }
                    }
                };
                if (s0 > SC) {
                    fill(NEGW);
                    continue;
                }
                int32_t d0;  // distance of slot s0
                if constexpr (MODE == 0) {
                    d0 = (int32_t)((c0 + t) % M) - (int32_t)s0;
                    while (d0 < 0) d0 += M;
                } else {
                    d0 = (int32_t)P - (int32_t)s0;
                }
                if (s0 + 31 < SC) {  // no sink column in the tile: whole-row fast paths
                    if constexpr (MODE == 0) {
                        const int32_t lim = (int32_t)(W < P + 1 ? W : P + 1);  // valid: d < lim
                        if (d0 >= 31) {
                            if (d0 < lim) {
                                run(d0);
                                continue;
                            }
                            if (d0 - 31 >= lim) {
                                fill(NEGW);
                                continue;
                            }
                        }
                    } else {
                        if (d0 < 0) {
                            fill(NEGW);
                            continue;
                        }
                        if (d0 - 31 >= (int32_t)W) {
                            fill(0);
                            continue;
                        }
                        if (d0 - 31 >= 0 && d0 < (int32_t)W) {
                            run(d0);
                            continue;
                        }
                    }
                }
                for (uint32_t c = 0; c < 32; ++c) {  // element by element
                    const uint32_t s = s0 + c;
                    uint16_t v;
                    if (s == SC) {
                        v = sink[rp];  // column 0 of the sink tile's row r
                    } else if (s > SC) {
                        v = NEG;
                    } else {
                        int32_t d = d0 - (int32_t)c;
                        if constexpr (MODE == 0) {
                            while (d < 0) d += M;
                            v = ((uint32_t)d < W && (uint32_t)d <= P) ? src[rp + ((L - 1 - d) >> 5) * 1024 +
                                                                           (((L - 1 - d) & 31) >> 4) * 256 + ((L - 1 - d) & 15)]
                                                                     : NEG;
                        } else {
                            v = d < 0 ? NEG
                                      : (d < (int32_t)W ? src[rp + ((L - 1 - d) >> 5) * 1024 + (((L - 1 - d) & 31) >> 4) * 256 +
                                                              ((L - 1 - d) & 15)]
                                                        : (uint16_t)0);
                        }
                    }
                    dst[rp + (c >> 4) * 256 + (c & 15)] = v;
                }
            }
            noc_async_write_barrier();
            noc_async_write(ob, sO.get_noc_addr((h * MTo + mo) * OT + jt), TB);
        }
    }
    noc_async_write_barrier();
}
