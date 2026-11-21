"""Paged KV for several conversations sharing one pool of physical blocks.

All attention layers (and the MTP layer) use the same logical -> physical block map, so one page table serves
every layer. The device page tables are fixed tensors (captured traces keep reading the same buffers); switching
conversations or growing one only rewrites their contents.
"""
import torch
import ttnn


class KVPager:
    def __init__(self, dev, nblocks, block_size=64):
        self.dev, self.nb, self.BS = dev, nblocks, block_size
        self.free = list(range(nblocks - 1, -1, -1))  # pop() hands out low blocks first
        self.blocks = []  # physical blocks of the active conversation, in logical order
        self.on_full = None  # callback(): free the blocks of some other conversation; returns False if none left
        ident = torch.arange(nblocks, dtype=torch.int32)
        self.pt1 = ttnn.from_torch(ident.view(1, nblocks), dtype=ttnn.int32, device=dev)
        self.ptT = {}

    def table_T(self, T):
        """[T, nblocks] copy of the table (decode SDPA with T "users")."""
        if T not in self.ptT:
            self.ptT[T] = ttnn.from_torch(torch.arange(self.nb, dtype=torch.int32).repeat(T, 1), dtype=ttnn.int32,
                                          device=self.dev)
        return self.ptT[T]

    def _write(self):
        row = torch.zeros(self.nb, dtype=torch.int32)  # unmapped logical blocks point at a valid block; never read
        if self.blocks:
            row[: len(self.blocks)] = torch.tensor(self.blocks, dtype=torch.int32)
            row[len(self.blocks):] = self.blocks[0]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(row.view(1, -1), dtype=ttnn.int32), self.pt1)
        for T, t in self.ptT.items():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(row.repeat(T, 1), dtype=ttnn.int32), t)

    def activate(self, blocks):
        """Make `blocks` (a list owned by the caller, grown in place by ensure) the active map."""
        self.blocks = blocks
        self._write()

    def ensure(self, end_pos):
        """Map logical blocks for positions [0, end_pos) of the active conversation."""
        need = min((end_pos + self.BS - 1) // self.BS, self.nb)
        if len(self.blocks) >= need:
            return
        while len(self.blocks) < need:
            if not self.free and not (self.on_full and self.on_full()):
                raise RuntimeError(f"KV pool exhausted ({self.nb} blocks)")
            self.blocks.append(self.free.pop())
        self._write()

    def release(self, blocks):
        self.free.extend(reversed(blocks))
        blocks.clear()

    def used(self):
        return self.nb - len(self.free)
