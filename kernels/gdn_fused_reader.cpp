// Fused GDN (post-conv) — reader. One value head h per loop iteration.
//   X  : conv output  [1,1,32,qkv_dim] bf16 tiles; q/k of key head kh=h/rep at kh*4+j / KOFF+kh*4+j, v at VOFF+h*4+j
//   Y  : in-proj output [1,1,32,Np] bf16; z of head h at ZOFF+h*4+j
//   GE : [1,1,32,2*NV] fp32; col h = exp(g) of head h (row = token)
//   BE : [1,1,32,2*NV] fp32; col NV+h = beta of head h
//   C  : constants fp32 tiles: 0 ones, 1 ones/128, 2 eps, 3..6 norm weight rows, 7.. E_t (T tiles)
//   S  : state snapshots [1,slots*NV,128,128] fp32 (slot-major); lane u reads slot IDX[u]
//   IDX: uint32 row-major tensor, element u = slot to read lane u's state from
// Lanes (batched decode): the T rows are T/SEG independent sequences of SEG rows each (lane u = rows u*SEG..);
// each lane starts from its own state. SEG == T: one sequence (the original kernel).
// CT: 0 T, 1 HPC, 2 rep, 3 KOFF, 4 VOFF, 5 ZOFF, 6 NV, 7 NC (const tiles), 8 SEG, then accessors X, Y, GE, BE, C, S, IDX
// RT: 0..6 addrs X, Y, GE, BE, C, S, IDX; 7 first head

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t c_SIN = 19, c_Q = 2, c_K = 3, c_V = 5, c_B = 6, c_G = 7, c_C = 8, c_Z = 12, c_SCR = 20;

template <typename Acc>
FORCE_INLINE void read_tiles(Noc& noc, const Acc& acc, uint32_t cb_id, uint32_t first, uint32_t n) {
    CircularBuffer cb(cb_id);
    const uint32_t tsz = get_tile_size(cb_id);
    cb.reserve_back(n);
    for (uint32_t i = 0; i < n; ++i) {
        noc.async_read(acc, cb, tsz, {.page_id = first + i}, {.offset_bytes = i * tsz});
    }
    noc.async_read_barrier();
    cb.push_back(n);
}

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {  // element index inside a 32x32 tile (4 faces)
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);
    constexpr uint32_t REP = get_compile_time_arg_val(2);
    constexpr uint32_t KOFF = get_compile_time_arg_val(3);
    constexpr uint32_t VOFF = get_compile_time_arg_val(4);
    constexpr uint32_t ZOFF = get_compile_time_arg_val(5);
    constexpr uint32_t NV = get_compile_time_arg_val(6);
    constexpr uint32_t NC = get_compile_time_arg_val(7);
    constexpr uint32_t SEG = get_compile_time_arg_val(8);
    constexpr uint32_t NU = T / SEG;
    constexpr auto aX = TensorAccessorArgs<9>();
    constexpr auto aY = TensorAccessorArgs<aX.next_compile_time_args_offset()>();
    constexpr auto aGB = TensorAccessorArgs<aY.next_compile_time_args_offset()>();
    constexpr auto aBE = TensorAccessorArgs<aGB.next_compile_time_args_offset()>();
    constexpr auto aC = TensorAccessorArgs<aBE.next_compile_time_args_offset()>();
    constexpr auto aS = TensorAccessorArgs<aC.next_compile_time_args_offset()>();
    constexpr auto aI = TensorAccessorArgs<aS.next_compile_time_args_offset()>();
    const auto sX = TensorAccessor(aX, get_arg_val<uint32_t>(0));
    const auto sY = TensorAccessor(aY, get_arg_val<uint32_t>(1));
    const auto sGB = TensorAccessor(aGB, get_arg_val<uint32_t>(2));
    const auto sBE = TensorAccessor(aBE, get_arg_val<uint32_t>(3));
    const auto sC = TensorAccessor(aC, get_arg_val<uint32_t>(4));
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(5));
    const auto sI = TensorAccessor(aI, get_arg_val<uint32_t>(6));
    const uint32_t h0 = get_arg_val<uint32_t>(7);

    Noc noc;
    read_tiles(noc, sC, c_C, 0, NC);

    CircularBuffer scr(c_SCR);
    scr.reserve_back(2);  // private scratch, never pushed
    const uint32_t scr_addr = scr.get_write_ptr();
    const uint32_t tsz32 = get_tile_size(c_SCR);

    // slot index (state after the last committed token of the previous step)
    noc.async_read(sI, scr, 64, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    uint32_t slots[NU];
    for (uint32_t u = 0; u < NU; ++u) {
        slots[u] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr)[u];
    }

    for (uint32_t hh = 0; hh < HPC; ++hh) {
        const uint32_t h = h0 + hh;
        const uint32_t kh = h / REP;
        read_tiles(noc, sX, c_K, KOFF + kh * 4, 4);
        read_tiles(noc, sS, c_SIN, (slots[0] * NV + h) * 16, 16);

        // exp(g) and beta scalars -> G_t tiles (full) and B tile (row t)
        const uint32_t gtile = h / 32, btile = (NV + h) / 32;
        noc.async_read(sGB, scr, tsz32, {.page_id = gtile}, {.offset_bytes = 0});
        noc.async_read(sBE, scr, tsz32, {.page_id = btile}, {.offset_bytes = tsz32});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
        {
            CircularBuffer cg(c_G);
            cg.reserve_back(T);
            volatile tt_l1_ptr uint32_t* gp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_write_ptr());
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t v = sp[elem_off(t, h % 32)];
                for (uint32_t e = 0; e < 1024; ++e) {
                    gp[t * 1024 + e] = v;
                }
            }
            cg.push_back(T);
        }
        {
            CircularBuffer cb(c_B);
            cb.reserve_back(1);
            volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
            for (uint32_t e = 0; e < 1024; ++e) {
                bp[e] = 0;
            }
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t vv = sp[1024 + elem_off(t, (NV + h) % 32)];  // second scratch tile
                for (uint32_t c = 0; c < 32; ++c) {
                    bp[elem_off(t, c)] = vv;
                }
            }
            cb.push_back(1);
        }
        read_tiles(noc, sX, c_V, VOFF + h * 4, 4);
        read_tiles(noc, sX, c_Q, kh * 4, 4);
        read_tiles(noc, sY, c_Z, ZOFF + h * 4, 4);
        // the other lanes' states: c_SIN holds one state; compute frees it after the lane's first token
        for (uint32_t u = 1; u < NU; ++u) {
            read_tiles(noc, sS, c_SIN, (slots[u] * NV + h) * 16, 16);
        }
    }
}
