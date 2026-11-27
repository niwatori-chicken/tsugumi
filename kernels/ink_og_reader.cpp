// Inkling decode: attention output gate (out = o * sigmoid(g), row 0 of every tile only) — reader. The core gathers
// row 0 of its n (<= 32) output tiles o0 .. o0 + n - 1 into a dense tile each for o (CB 1) and g (CB 0): row r =
// row 0 of O tile o0 + r / G tile goff + o0 + r (G: 32-byte face rows, must be in L1; O: whole tiles, L1 or DRAM).
// CT: accessors O, G.  RT: 0 o, 1 g, 2 o0, 3 n, 4 goff (column tile of the gate in G)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aO = TensorAccessorArgs<0>();
    constexpr auto aG = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    const auto sO = TensorAccessor(aO, get_arg_val<uint32_t>(0));
    const auto sG = TensorAccessor(aG, get_arg_val<uint32_t>(1));
    const uint32_t o0 = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3), goff = get_arg_val<uint32_t>(4);
    Noc noc;
    CircularBuffer cg(0), co(1), cs(2);
    cg.reserve_back(1);
    co.reserve_back(1);
    if (n < 32) {
        volatile tt_l1_ptr uint32_t* zg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_write_ptr());
        volatile tt_l1_ptr uint32_t* zo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(co.get_write_ptr());
        for (uint32_t i = 0; i < 512; ++i) zg[i] = zo[i] = 0;
    }
    // O may be in DRAM (64-byte alignment): whole tiles into the scratch CB, then the row-0 face rows copied locally
    cs.reserve_back(n);
    const uint32_t sb = cs.get_write_ptr();
    for (uint32_t r = 0; r < n; ++r) noc.async_read(sO, cs, 2048, {.page_id = o0 + r}, {.offset_bytes = r * 2048});
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t d0 = (r >> 4) * 1024 + (r & 15) * 32;
        for (uint32_t h = 0; h < 2; ++h)  // G is in L1 (32-byte face rows)
            noc.async_read(sG, cg, 32, {.page_id = goff + o0 + r, .offset_bytes = h * 512}, {.offset_bytes = d0 + h * 512});
    }
    noc.async_read_barrier();
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t d0 = (r >> 4) * 1024 + (r & 15) * 32;
        for (uint32_t h = 0; h < 2; ++h) {
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(co.get_write_ptr() + d0 + h * 512);
            volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sb + r * 2048 + h * 512);
            for (uint32_t i = 0; i < 8; ++i) d[i] = q[i];
        }
    }
    cg.push_back(1);
    co.push_back(1);
}
