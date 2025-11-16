#include "definitions.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include "avx.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <cstdio>

using namespace frequency_map;

FrequencyMap frequency_map::createFrequencyMap(char* input)
{
	FrequencyMap ret = {};
	while (*input) {
		char c = *input;
		*input++;
		if (c >= 'a' && c <= 'z') {
			c = c - 'a' + 'A';
		}
		if (c < 'A' || c > 'Z') {
			throw new std::runtime_error("unspecified");
		}
		ret[c - 'A']++;
	}
	return ret;
}
FrequencyMap frequency_map::createFrequencyMap(string input)
{
	return createFrequencyMap(input.data());
}
