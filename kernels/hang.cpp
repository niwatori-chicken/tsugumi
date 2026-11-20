// Test-only kernel for hang detection (scripts/test_hang.py). RT 0: spin flag.
// With spin = 1 it waits forever on an L1 word nobody writes, so the device never finishes the program.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    if (get_arg_val<uint32_t>(0) == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(1));
    while (p[0] != 0xdeadbeef) {
    }
}
