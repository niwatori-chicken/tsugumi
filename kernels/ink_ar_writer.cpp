// Inkling decode: block attention residual + rms norm — writer.
// phase 1: the Gram partial (CB 8, float32) -> PART page c.
// phase 2: from G (CB 8): f = rsqrt(G[nb][nb] / H + eps) (rms of the raw h), for query k (row 31 - k):
//   s_j = G[31-k][j] * (j == nb ? f : 1), p = softmax(s_0..s_nb), w_j = p_j * (j == nb ? f : 1),
//   r_k = rsqrt(w^T G w / H + eps)  (rms of the output);  weight tiles (bf16) -> CB 9: W_0..W_NQ-1 (row 0 = w),
//   W2_k (row 0 = w * r_k), FD (diag (1, .., 1, f)); then the outputs: CB 16 -> AI_k, CB 17 -> XN_k, CB 18 -> TOUT
//   (page j0 + j of each).
// CT: accessors PART (L1), OUT (L1), TOUT.
// RT: 0 phase, 1 part, 2 c, 3 nb, 4 NQ, 5 use_ln, 6 tout, 7 inv_H, 8 eps (float bits), 9..11 ai0..2, 12..14 xn0..2,
//     15 tout addr, 16 j0, 17 NT
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE uint32_t elem_off(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE float fl(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

FORCE_INLINE uint16_t bf(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

float rsq(float x) {
    uint32_t i;
    __builtin_memcpy(&i, &x, 4);
    i = 0x5f3759dfu - (i >> 1);
    float y = fl(i);
    for (int it = 0; it < 4; ++it) y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

float ex(float x) {  // exp(x), x <= 0
    if (x < -87.0f) return 0.0f;
    const float y = x * 1.4426950408889634f;
    int k = (int)y;
    if ((float)k > y) --k;
    const float z = (y - (float)k) * 0.6931471805599453f;
    float p = 1.0f + z / 8.0f;
    p = 1.0f + z / 7.0f * p;
    p = 1.0f + z / 6.0f * p;
    p = 1.0f + z / 5.0f * p;
    p = 1.0f + z / 4.0f * p;
    p = 1.0f + z / 3.0f * p;
    p = 1.0f + z / 2.0f * p;
    p = 1.0f + z * p;
    return p * fl((uint32_t)(k + 127) << 23);
}

void kernel_main() {
    constexpr auto aP = TensorAccessorArgs<0>();
    constexpr auto aO = TensorAccessorArgs<aP.next_compile_time_args_offset()>();
    constexpr auto aT = TensorAccessorArgs<aO.next_compile_time_args_offset()>();
    const uint32_t phase = get_arg_val<uint32_t>(0);
    Noc noc;
    CircularBuffer cg(8);
    if (phase == 1) {
        const auto sP = TensorAccessor(aP, get_arg_val<uint32_t>(1));
        cg.wait_front(1);
        noc.async_write(cg, sP, 4096, {.offset_bytes = 0}, {.page_id = get_arg_val<uint32_t>(2)});
        noc.async_write_barrier();
        cg.pop_front(1);
        return;
    }
    const uint32_t nb = get_arg_val<uint32_t>(3), NQ = get_arg_val<uint32_t>(4), use_ln = get_arg_val<uint32_t>(5);
    const uint32_t tout = get_arg_val<uint32_t>(6);
    const float inv_h = fl(get_arg_val<uint32_t>(7)), eps = fl(get_arg_val<uint32_t>(8));
    const uint32_t j0 = get_arg_val<uint32_t>(16), NT = get_arg_val<uint32_t>(17);
    cg.wait_front(1);
    volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_read_ptr());
    const uint32_t n = nb + 1;
    float g[5][5];
    for (uint32_t a = 0; a < n; ++a)
        for (uint32_t b = 0; b < n; ++b) g[a][b] = fl(G[elem_off(a, b)]);
    const float f = rsq(g[nb][nb] * inv_h + eps);
    float w[3][5], r[3];
    for (uint32_t k = 0; k < NQ; ++k) {
        float s[5], m = -1e30f, sum = 0.0f;
        for (uint32_t j = 0; j < n; ++j) {
            s[j] = fl(G[elem_off(31 - k, j)]) * (j == nb ? f : 1.0f);
            if (s[j] > m) m = s[j];
        }
        for (uint32_t j = 0; j < n; ++j) {
            s[j] = ex(s[j] - m);
            sum += s[j];
        }
        const float inv = 1.0f / sum;
        for (uint32_t j = 0; j < n; ++j) w[k][j] = s[j] * inv * (j == nb ? f : 1.0f);
        float q = 0.0f;
        for (uint32_t a = 0; a < n; ++a)
            for (uint32_t b = 0; b < n; ++b) q += w[k][a] * w[k][b] * g[a][b];
        r[k] = rsq(q * inv_h + eps);
    }
    cg.pop_front(1);
    CircularBuffer cw(9);
    const uint32_t nw = 2 * NQ + 1;
    cw.reserve_back(nw);
    const uint32_t base = cw.get_write_ptr();
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for (uint32_t i = 0; i < nw * 512; ++i) z[i] = 0;
    volatile tt_l1_ptr uint16_t* W = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    for (uint32_t k = 0; k < NQ; ++k) {
        for (uint32_t j = 0; j < n; ++j) {
            W[k * 1024 + elem_off(0, j)] = bf(w[k][j]);
            W[(NQ + k) * 1024 + elem_off(0, j)] = bf(w[k][j] * r[k]);
        }
    }
    for (uint32_t j = 0; j < n; ++j) W[2 * NQ * 1024 + elem_off(j, j)] = bf(j == nb ? f : 1.0f);
    cw.push_back(nw);
    CircularBuffer cai(16), cxn(17), cto(18);
    for (uint32_t j = 0; j < NT; ++j) {
        for (uint32_t k = 0; k < NQ; ++k) {
            // with norm weights only XN_k is written, else only AI_k
            CircularBuffer& co = use_ln ? cxn : cai;
            const auto sA = TensorAccessor(aO, get_arg_val<uint32_t>((use_ln ? 12 : 9) + k));
            co.wait_front(1);
            noc.async_write(co, sA, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
            noc.async_write_barrier();
            co.pop_front(1);
        }
        if (tout) {
            const auto sT = TensorAccessor(aT, get_arg_val<uint32_t>(15));
            cto.wait_front(1);
            noc.async_write(cto, sT, 2048, {.offset_bytes = 0}, {.page_id = j0 + j});
            noc.async_write_barrier();
            cto.pop_front(1);
        }
    }
}
