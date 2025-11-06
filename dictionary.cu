#include "definitions.cuh"
#include "dictionary.cuh"
#include "avx.hpp"
#include <stdint.h>

using namespace dictionary;

frequency_map::Result dictionary::Dictionary::d_compareFrequencyMaps_pip(
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
	frequency_map::Result ret = complete_match;
	for (int8_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
		if ((output[i] = input[i] - other[i]) < 0) {
			return no_match;
		}
		int8_t r = (output[i] = input[i] - other[i]);
		if (r < 0) {
			return no_match;
		}
		else if (r > 0) {
			ret = incomplete_match;
		}
	}
	return ret;
}

frequency_map::Result dictionary::Dictionary::h_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
    if (other == NULL) {
		throw;
	}
	frequency_map::Result ret = complete_match;
	auto ret = avx::compare(
		input,
		other,
		output
	);
	if (ret.any_negative) {
		return no_match;
	}
	return !ret.all_zero ? incomplete_match : complete_match;
}
