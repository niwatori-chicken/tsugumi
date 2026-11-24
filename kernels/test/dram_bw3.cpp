// MoE weight streaming pattern test (no compute): core c reads, for 8 experts, its two halves of HT tiles from
// banks c % 4 (+ 4) in blocks of BT tiles with up to NF blocks in flight (rotating transaction ids).
// RT: 0 base, 1 c, 2 HT, 3 BT, 4 NF, 5 stagger (0/1), 6.. expert ids (8), 14 single bank (0/1)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1), HT = get_arg_val<uint32_t>(2);
    const uint32_t BT = get_arg_val<uint32_t>(3), NF = get_arg_val<uint32_t>(4), stagger = get_arg_val<uint32_t>(5);
    const uint32_t single = get_arg_val<uint32_t>(14);
    constexpr uint32_t pb = 1088;
    const uint32_t l1 = get_write_ptr(0);
    const uint32_t first = stagger ? (c / 4) % 2 : 0;
    uint32_t n = 0;
    for (uint32_t k = 0; k < 8; ++k) {
        const uint32_t e = get_arg_val<uint32_t>(6 + k);
        for (uint32_t hh = 0; hh < 2; ++hh) {
            const uint32_t h = hh ^ first;
            const uint32_t bank = single ? (c % 8) : (c % 4 + 4 * h);
            const uint32_t slot = single ? (c / 8) * 2 + h : c / 4;
            const uint32_t a = base + (e * 9 + slot) * HT * pb;
            for (uint32_t g = 0; g < HT / BT; ++g) {
                if (n >= NF) noc_async_read_barrier_with_trid(1 + (n - NF) % 8);
                noc_async_read_set_trid(1 + n % 8);
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, a + g * BT * pb), l1 + (n % NF) * BT * pb, BT * pb);
                ++n;
            }
        }
    }
    noc_async_read_barrier();
}
