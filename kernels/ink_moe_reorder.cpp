// Inkling MoE: rearrange interleaved expert weights (bfp8 pages) into the bank-contiguous layout read by
// ink_moe_reader.cpp. Work item (e, q), q = 2 c + h (core c of the MoE program, half h of its K range): T tiles
// to DRAM bank c % 4 + 4 h at tile offset ((e * 9 + c / 4) * T) of the destination buffer.
//   kind 0 (gate_up [E, 72, 72] tiles): tile i -> kt = h * 36 + i / 2, nt = c + 36 * (i % 2)       (T = 72)
//   kind 1 (down [E, 36, 72] tiles):    tile i -> kt = h * 18 + i / 2, nt = 2 c + i % 2             (T = 36)
// CB 0: T pages.  CT: accessor SRC.  RT: 0 src, 1 dst, 2 kind, 3 first item, 4 item stride, 5 items, 6 page bytes
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto aS = TensorAccessorArgs<0>();
    const auto sS = TensorAccessor(aS, get_arg_val<uint32_t>(0));
    const uint32_t dst = get_arg_val<uint32_t>(1), kind = get_arg_val<uint32_t>(2);
    const uint32_t it0 = get_arg_val<uint32_t>(3), stride = get_arg_val<uint32_t>(4), nit = get_arg_val<uint32_t>(5);
    const uint32_t pb = get_arg_val<uint32_t>(6);
    const uint32_t T = kind == 0 ? 72 : 36;
    Noc noc;
    CircularBuffer cb(0);
    cb.reserve_back(T);
    const uint32_t l1 = cb.get_write_ptr();
    for (uint32_t it = it0; it < nit; it += stride) {
        const uint32_t e = it / 72, q = it % 72, c = q / 2, h = q % 2;
        for (uint32_t i = 0; i < T; ++i) {
            uint32_t page;
            if (kind == 0) {
                page = (e * 72 + h * 36 + i / 2) * 72 + c + 36 * (i % 2);
            } else {
                page = (e * 36 + h * 18 + i / 2) * 72 + 2 * c + i % 2;
            }
            noc.async_read(sS, cb, pb, {.page_id = page}, {.offset_bytes = i * pb});
        }
        noc.async_read_barrier();
        const uint32_t bank = c % 4 + 4 * h, slot = c / 4;  // banks b and b + 4 are near the same worker rows
        const uint32_t base = dst + (e * 9 + slot) * T * pb;
        for (uint32_t i = 0; i < T; i += 8) {
            const uint32_t n = (T - i) < 8 ? (T - i) : 8;
            noc_async_write(l1 + i * pb, get_noc_addr_from_bank_id<true>(bank, base + i * pb), n * pb);
        }
        noc_async_write_barrier();
    }
}
