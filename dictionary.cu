#include "definitions.cuh"
#include "dictionary.cuh"
#include "avx.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string.h>

using namespace dictionary;

__device__ frequency_map::Result dictionary::Dictionary::d_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
	if (other == NULL) {
		printf("other_index %d found NULL getFrequencyMapPointer", other_index);
		__trap();
	}
	frequency_map::Result ret = COMPLETE_MATCH;
	for (int8_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
		int8_t r = ((*output)[i] = (*input)[i] - (*other)[i]);
		if (r < 0) {
			return NO_MATCH;
		}
		else if (r > 0) {
			ret = INCOMPLETE_MATCH;
		}
	}
	return ret;
}

__host__ frequency_map::Result dictionary::Dictionary::h_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
    if (other == NULL) {
		throw;
	}
	auto ret = avx::compare(
		*input,
		*other,
		(int8_t*)output
	);
	if (ret.any_negative) {
		return NO_MATCH;
	}
	return !ret.all_zero ? INCOMPLETE_MATCH : COMPLETE_MATCH;
}

frequency_map::FrequencyMap* dictionary::Dictionary::getFrequencyMapPointer(
	FrequencyMapIndex_t index
)
{
	return nullptr;
}
