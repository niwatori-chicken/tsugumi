// DRAM read bandwidth test, bank-contiguous: read BYTES from DRAM bank BANK starting at ADDR + OFF in CHUNK-byte
// reads, NB in flight (CB 0 holds NB chunks). RT: 0 addr, 1 bank, 2 off, 3 bytes, 4 chunk, 5 NB
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0), bank = get_arg_val<uint32_t>(1), off = get_arg_val<uint32_t>(2);
    const uint32_t bytes = get_arg_val<uint32_t>(3), chunk = get_arg_val<uint32_t>(4), NB = get_arg_val<uint32_t>(5);
    const uint32_t l1 = get_write_ptr(0);
    uint32_t slot = 0, n = 0;
    for (uint32_t o = 0; o < bytes; o += chunk) {
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank, addr + off + o);
        noc_async_read(src, l1 + slot * chunk, chunk);
        if (++slot == NB) slot = 0;
        if (++n % NB == 0) noc_async_read_barrier();
    }
    noc_async_read_barrier();
}
