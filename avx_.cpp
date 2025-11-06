#include <immintrin.h>
#include <cstddef>
#include <cstdint>

using namespace avx;

// a, b, dst each point to 26-byte, 32B-aligned arrays (int8_t == signed char).
// Returns true if any (a[i]-b[i]) < 0.
static inline bool avx::sub_and_any_neg_i8_26(
		__restrict const int8_t* a, __restrict const int8_t* b, __restrict int8_t* dst) {
    // mask for 8 dword lanes: enable first 6 lanes (24 bytes), disable last 2 lanes.
    const __m256i mask24 = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);

    // Load first 24 bytes of a/b (masked), zero the rest of the vector.
    const __m256i va24 = _mm256_maskload_epi32(reinterpret_cast<const int*>(a), mask24);
    const __m256i vb24 = _mm256_maskload_epi32(reinterpret_cast<const int*>(b), mask24);

    // Vector subtract on bytes.
    const __m256i vr24   = _mm256_sub_epi8(va24, vb24);

    // Compare (vr24 < 0) per signed byte.
    const __m256i vzero  = _mm256_setzero_si256();
    const __m256i vneg24 = _mm256_cmpgt_epi8(vzero, vr24);  // 0xFF where negative

    // Any negatives in first 24 bytes?
    int any_mask = _mm256_movemask_epi8(vneg24);            // bit per byte
    // Only the low 24 bits can be set because of the maskload zeros above.

    // Store first 24 result bytes (masked store => writes only those 24 bytes).
    _mm256_maskstore_epi32(reinterpret_cast<int*>(dst), mask24, vr24);

    // Tail: bytes 24,25
    const int8_t r24 = static_cast<int8_t>(a[24] - b[24]);
    const int8_t r25 = static_cast<int8_t>(a[25] - b[25]);
    dst[24] = r24; dst[25] = r25;

    const bool any_tail = (r24 < 0) | (r25 < 0);
    return (any_mask != 0) | any_tail;
}

