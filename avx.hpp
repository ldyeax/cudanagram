#pragma once
#include <utility>
#include <cstdint>
//#include <immintrin.h>
namespace avx {
	struct Result {
		bool any_negative;
		bool all_zero;
	};

	/**
	 * Subtract b from a, store in dst
	 * **/
	static inline Result sub_and_flags_i8_26 (
		__restrict const int8_t* a,
		__restrict const int8_t* b,
		__restrict int8_t* dst
	);

	/**
	 * Subtract b from a and return result
	 * **/
	Result compare(
		int8_t* a,
		int8_t* b
	);

	/**
	 * Subtract b from a, store in dst
	 * **/
	Result compare(
		int8_t* a,
		int8_t* b,
		int8_t* dst
	);
}
