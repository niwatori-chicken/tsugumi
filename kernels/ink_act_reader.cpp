// Inkling decode MoE: softcapped SwiGLU of the gate_up output (row 0 of every tile only) — reader.
// Output tile o (expert slot k = o / IT, column tile t = o % IT) needs row 0 of GU pages k * 2IT + t (gate) and
// k * 2IT + IT + t (up). The core gathers the row 0s of its 32 output tiles o0 .. o0 + 31 into one dense tile each
// for gate and up: row r = row 0 of tile o0 + r (32-byte face rows; GU must be in L1).
// CB 0 gate, 1 up (1 tile each).  CT: accessor GU.  RT: 0 gu, 1 o0, 2 n (<= 32), 3 IT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aG = TensorAccessorArgs<0>();
    const auto sG = TensorAccessor(aG, get_arg_val<uint32_t>(0));
    const uint32_t o0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2), IT = get_arg_val<uint32_t>(3);
    Noc noc;
    CircularBuffer cg(0), cu(1);
    cg.reserve_back(1);
    cu.reserve_back(1);
    volatile tt_l1_ptr uint32_t* zg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_write_ptr());
    volatile tt_l1_ptr uint32_t* zu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cu.get_write_ptr());
    if (n < 32)
        for (uint32_t i = 0; i < 512; ++i) zg[i] = zu[i] = 0;
    for (uint32_t r = 0; r < n; ++r) {
        const uint32_t o = o0 + r, k = o / IT, t = o % IT;
        const uint32_t d0 = (r >> 4) * 1024 + (r & 15) * 32;  // face 0 / 2 row
        for (uint32_t h = 0; h < 2; ++h) {
            noc.async_read(sG, cg, 32, {.page_id = k * 2 * IT + t, .offset_bytes = h * 512},
                           {.offset_bytes = d0 + h * 512});
            noc.async_read(sG, cu, 32, {.page_id = k * 2 * IT + IT + t, .offset_bytes = h * 512},
                           {.offset_bytes = d0 + h * 512});
        }
    }
    noc.async_read_barrier();
    cg.push_back(1);
    cu.push_back(1);
}
