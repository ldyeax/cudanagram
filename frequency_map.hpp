#pragma once
#include "definitions.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <cstdio>

using std::string;

namespace frequency_map {
	struct FrequencyMap {
		char frequencies[NUM_LETTERS_IN_ALPHABET];
		// __device__ __host__ operator int8_t*()
		// {
		// 	return (int8_t*)&frequencies[0];
		// }
		__device__ __host__ operator int8_t*()
		{
			return (int8_t*)&frequencies[0];
		}
		__device__ __host__ const std::byte* asStdBytePointer() const
		{
			return (const std::byte*)&frequencies[0];
		}

		void print()
		{
			for (int32_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
				fprintf(stderr, "%c=%d ", i+'A', frequencies[i]);
			}
			fprintf(stderr, "\n");
		}

		__device__ void d_print()
		{
			for (int32_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
				printf("%c=%d ", i+'A', frequencies[i]);
			}
			printf("\n");
		}

		bool isAllZero()
		{
			for (int32_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
				if (frequencies[i] != 0) {
					return false;
				}
			}
			return true;
		}

		bool anyNegative()
		{
			for (int32_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
				if (frequencies[i] < 0) {
					return true;
				}
			}
			return false;
		}

		bool equals(FrequencyMap& other)
		{
			for (int32_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
				if (frequencies[i] != other.frequencies[i]) {
					return false;
				}
			}
			return true;
		}
	};
	enum Result {
		no_match = 0,
		incomplete_match = 1,
		complete_match = 3
	};
	FrequencyMap createFrequencyMap(char* input);
	FrequencyMap createFrequencyMap(string input);
}
