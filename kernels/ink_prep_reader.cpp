// Inkling decode: per-head preparation of q / r / k / v of one token — reader. One core per role; a role takes
// NROW heads of HW values from row 0 of a linear's output (SRC, starting at column tile SOFF) into NT tiles with
// head h in row h (32-byte face-row reads), plus, for the convolved roles (k / v), the 3 history rows of those
// channels from HIST (row r = lag r, rows read and written as 64-byte pairs: DRAM alignment), the 4 tap tiles per
// lag (TAPS, already in head-row layout), and the dynamic taps (DYN: 32 values head * 4 + lag at column tile DOFF of
// row 0) as column tiles. The history is shifted here: rows 1..3 <- the current input and lags 1, 2 (row 0 = 0).
// CB: 0 X (lag 0) NT, 1 H (lags 1..3) 3 NT, 2 TAPS 4 NT, 3 DYN 4, 4 SCALE NT, 5 TAU 1, 6 SC (reduce scaler) 1,
//     20 scratch
// CT: accessors L1 (SRC / DYN, interleaved L1 bf16 tiles), DRAM (HIST / TAPS / SCALE / TAU / SC)
// RT: 0 src, 1 soff, 2 nrow, 3 hw, 4 nt, 5 conv, 6 hist, 7 hoff, 8 taps, 9 toff, 10 dyn, 11 dsrc, 12 doff,
//     13 scale, 14 scale addr, 15 tau, 16 tau addr, 17 sc addr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_X = 0, c_H = 1, c_T = 2, c_D = 3, c_S = 4, c_TAU = 5, c_SC = 6, c_SCR = 20;

FORCE_INLINE uint32_t face_off(uint32_t row, uint32_t f) { return (((row >> 4) * 2 + f) * 256 + (row & 15) * 16) * 2; }

FORCE_INLINE void zero(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) z[i] = 0;
}

FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < 8; ++i) d[i] = s[i];
}

void kernel_main() {
    constexpr auto aL = TensorAccessorArgs<0>();
    constexpr auto aD = TensorAccessorArgs<aL.next_compile_time_args_offset()>();
    const uint32_t nrow = get_arg_val<uint32_t>(2), hw = get_arg_val<uint32_t>(3), nt = get_arg_val<uint32_t>(4);
    const auto sSrc = TensorAccessor(aL, get_arg_val<uint32_t>(0));
    const uint32_t soff = get_arg_val<uint32_t>(1);
    const uint32_t conv = get_arg_val<uint32_t>(5), dyn = get_arg_val<uint32_t>(10);
    const uint32_t scale = get_arg_val<uint32_t>(13), tau = get_arg_val<uint32_t>(15);
    constexpr uint32_t TB = 2048;
    Noc noc;

    CircularBuffer sc(c_SC);
    sc.reserve_back(1);
    noc.async_read(TensorAccessor(aD, get_arg_val<uint32_t>(17)), sc, TB, {.page_id = 0}, {.offset_bytes = 0});

    // lag 0 (the current token): head h -> row h
    CircularBuffer cx(c_X);
    cx.reserve_back(nt);
    const uint32_t xb = cx.get_write_ptr();
    zero(xb, nt * TB);
    for (uint32_t h = 0; h < nrow; ++h) {
        for (uint32_t d = 0; d < nt; ++d) {
            if (hw >= 32) {  // a head spans whole source tiles
                for (uint32_t f = 0; f < 2; ++f) {
                    noc.async_read(sSrc, cx, 32, {.page_id = soff + h * nt + d, .offset_bytes = face_off(0, f)},
                                   {.offset_bytes = d * TB + face_off(h, f)});
                }
            } else {  // hw == 16: one face row
                const uint32_t col = h * hw;
                noc.async_read(sSrc, cx, 32, {.page_id = soff + col / 32, .offset_bytes = face_off(0, (col / 16) & 1)},
                               {.offset_bytes = face_off(h, 0)});
            }
        }
    }
    if (scale) {
        CircularBuffer cs(c_S);
        cs.reserve_back(nt);
        const auto sS = TensorAccessor(aD, get_arg_val<uint32_t>(14));
        for (uint32_t d = 0; d < nt; ++d) {
            noc.async_read(sS, cs, TB, {.page_id = d}, {.offset_bytes = d * TB});
        }
        noc.async_read_barrier();
        cs.push_back(nt);
    }
    if (tau) {
        CircularBuffer ct(c_TAU);
        ct.reserve_back(1);
        noc.async_read(TensorAccessor(aD, get_arg_val<uint32_t>(16)), ct, TB, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        ct.push_back(1);
    }
    if (conv) {
        const auto sH = TensorAccessor(aD, get_arg_val<uint32_t>(6));
        const uint32_t hoff = get_arg_val<uint32_t>(7);
        const auto sT = TensorAccessor(aD, get_arg_val<uint32_t>(8));
        const uint32_t toff = get_arg_val<uint32_t>(9);
        CircularBuffer ch(c_H), ctp(c_T), cscr(c_SCR);
        ctp.reserve_back(4 * nt);
        for (uint32_t i = 0; i < 4 * nt; ++i) {
            noc.async_read(sT, ctp, TB, {.page_id = toff + i}, {.offset_bytes = i * TB});
        }
        // history rows (pairs 0-1 and 2-3 of each face, 128 bytes per (h, d, f)) -> scratch
        cscr.reserve_back(4);
        const uint32_t sb = cscr.get_write_ptr();
        for (uint32_t h = 0; h < nrow; ++h) {
            for (uint32_t d = 0; d < nt; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t o = ((h * nt + d) * 2 + f) * 128;
                    noc.async_read(sH, cscr, 64, {.page_id = hoff + h * nt + d, .offset_bytes = face_off(0, f)},
                                   {.offset_bytes = o});
                    noc.async_read(sH, cscr, 64, {.page_id = hoff + h * nt + d, .offset_bytes = face_off(2, f)},
                                   {.offset_bytes = o + 64});
                }
            }
        }
        noc.async_read_barrier();  // also lag 0 (cx) and the taps
        ch.reserve_back(3 * nt);
        const uint32_t hb = ch.get_write_ptr();
        zero(hb, 3 * nt * TB);
        for (uint32_t h = 0; h < nrow; ++h) {
            for (uint32_t d = 0; d < nt; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t o = sb + ((h * nt + d) * 2 + f) * 128;
                    copy32(hb + (0 * nt + d) * TB + face_off(h, f), o + 32);  // lag 1 = hist row 1
                    copy32(hb + (1 * nt + d) * TB + face_off(h, f), o + 64);  // lag 2 = row 2
                    copy32(hb + (2 * nt + d) * TB + face_off(h, f), o + 96);  // lag 3 = row 3
                    // new history: row 0 = 0, row 1 = lag 0, row 2 = lag 1, row 3 = lag 2
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o);
                    for (uint32_t i = 0; i < 8; ++i) z[i] = 0;
                    copy32(o + 96, o + 64);
                    copy32(o + 64, o + 32);
                    copy32(o + 32, xb + d * TB + face_off(h, f));
                }
            }
        }
        for (uint32_t h = 0; h < nrow; ++h) {
            for (uint32_t d = 0; d < nt; ++d) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t o = ((h * nt + d) * 2 + f) * 128;
                    noc.async_write(cscr, sH, 64, {.offset_bytes = o},
                                    {.page_id = hoff + h * nt + d, .offset_bytes = face_off(0, f)});
                    noc.async_write(cscr, sH, 64, {.offset_bytes = o + 64},
                                    {.page_id = hoff + h * nt + d, .offset_bytes = face_off(2, f)});
                }
            }
        }
        if (dyn) {  // dynamic taps: column tile per lag, row h = DYN[h * 4 + lag]
            const auto sDy = TensorAccessor(aL, get_arg_val<uint32_t>(11));
            const uint32_t doff = get_arg_val<uint32_t>(12);
            CircularBuffer cd(c_D);
            cd.reserve_back(4);
            const uint32_t db = cd.get_write_ptr();
            zero(db, 4 * TB);
            const uint32_t tmp = sb + nrow * nt * 2 * 128;
            for (uint32_t f = 0; f < 2; ++f) {
                noc.async_read(sDy, cscr, 32, {.page_id = doff, .offset_bytes = face_off(0, f)},
                               {.offset_bytes = nrow * nt * 2 * 128 + f * 32});
            }
            noc.async_read_barrier();
            volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
            for (uint32_t h = 0; h < nrow; ++h) {
                for (uint32_t lag = 0; lag < 4; ++lag) {
                    volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(db + lag * TB);
                    t[face_off(h, 0) / 2] = v[h * 4 + lag];
                }
            }
            cd.push_back(4);
        }
        noc.async_write_barrier();
        ch.push_back(3 * nt);
        ctp.push_back(4 * nt);
    } else {
        noc.async_read_barrier();
    }
    sc.push_back(1);
    cx.push_back(nt);
}
