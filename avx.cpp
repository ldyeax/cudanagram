#include <utility>
#include <cstdint>
#include <immintrin.h>

#include "avx.hpp"

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static inline avx::Result avx::sub_and_flags_i8_26 (
	const int8_t* a,
	const int8_t* b,
	int8_t* dst
)
{

    // Same vector ops as before
    const __m256i mask24 = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    const __m256i va24   = _mm256_maskload_epi32(reinterpret_cast<const int*>(a), mask24);
    const __m256i vb24   = _mm256_maskload_epi32(reinterpret_cast<const int*>(b), mask24);
    const __m256i vr24   = _mm256_sub_epi8(va24, vb24);

    const __m256i vzero  = _mm256_setzero_si256();
    const __m256i vneg24 = _mm256_cmpgt_epi8(vzero, vr24);
    const int any_mask24 = _mm256_movemask_epi8(vneg24);

    _mm256_maskstore_epi32(reinterpret_cast<int*>(dst), mask24, vr24);

    // Tail (2 bytes)
    const int8_t r24 = static_cast<int8_t>(a[24] - b[24]);
    const int8_t r25 = static_cast<int8_t>(a[25] - b[25]);
    dst[24] = r24; dst[25] = r25;

    const bool any_negative = (any_mask24 != 0) || (r24 < 0) || (r25 < 0);

    // SIMD “all-zero?” for the first 24 bytes, no new compares:
    // testz(vr24, vr24) => ZF=1 iff all bits of vr24 are zero
    const bool all_zero_vec24 = _mm256_testz_si256(vr24, vr24) != 0;

    // Final all-zero includes tail
    const bool all_zero = all_zero_vec24 && (r24 == 0) && (r25 == 0);

    Result ret;
	ret.any_negative = any_negative;
	ret.all_zero = all_zero;
	return ret;
}

avx::Result avx::compare(
	int8_t* a,
	int8_t* b,
	int8_t* dst
)
{
	return avx::sub_and_flags_i8_26(
		static_cast<const int8_t*>(a),
		static_cast<const int8_t*>(b),
		dst
	);
}

