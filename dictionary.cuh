#include "definitions.cuh"
#include <stdint.h>
#include <cstdint>

#define FrequencyMapIndex_t int32_t

namespace dictionary {
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
		__device__ __host__ frequency_map::FrequencyMap* getFrequencyMapPointer(
			FrequencyMapIndex_t other_index
		);
	};
}
