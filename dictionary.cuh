#include <stdint.h>
#include "definitions.cuh"

#define FrequencyMapIndex_t int32_t

namespace dictionary {
	namespace frequency_map {
		struct FrequencyMap {
			char frequencies[NUM_LETTERS_IN_ALPHABET];
		};
		enum Result {
			no_match = 0,
			incomplete_match = 1,
			complete_match = 3
		};
	}
	class Dictionary {
		__device__ frequency_map::Result d_compareFrequencyMaps_pip(
			frequency_map::FrequencyMap* input,
			FrequencyMapIndex_t other_index,
			frequency_map::FrequencyMap* output
		);
		__host__ frequency_map::Result h_compareFrequencyMaps_pip(
			frequency_map::FrequencyMap* input,
			FrequencyMapIndex_t other_index,
			frequency_map::FrequencyMap* output
		);
		__device__ __host__ getFrequencyMapPointer(
			FrequencyMapIndex_t other_index
		);
	}
}
