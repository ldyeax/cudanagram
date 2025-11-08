#pragma once
#include "definitions.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string>
#include <vector>

using std::string;

namespace frequency_map {
	struct FrequencyMap {
		char frequencies[NUM_LETTERS_IN_ALPHABET];
		__device__ __host__ operator int8_t*() {
			return (int8_t*)&frequencies[0];
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
