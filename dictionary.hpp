#include "definitions.hpp"
#include "frequency_map.hpp"
#include <stdint.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
using std::cout;
using std::vector;
using std::string;

#define FrequencyMapIndex_t int32_t

namespace dictionary {
	struct dictstats {
	    int32_t initial_words_parsed = 0;
	    int32_t initial_words_removed = 0;
		int32_t frequency_map_rejections = 0;
	};

	class Dictionary {
	private:
		void init();
	public:
		Dictionary(
			char* p_input,
			char* p_filename,
			char* p_buffer,
			int32_t p_buffer_length
		);
		string input = "";
		frequency_map::FrequencyMap input_frequency_map = {};
		frequency_map::FrequencyMap* frequency_maps = NULL;
		FrequencyMapIndex_t frequency_maps_length = 0;
		vector<int32_t>* word_index_lists_for_frequency_maps = NULL;
		char* buffer = NULL;
		int32_t buffer_length = 0;
		vector<string> words{};
		dictstats stats{};

		void printStats();

		FrequencyMapIndex_t getOrCreateFrequencyMapIndexByWordIndex(
			int32_t w_i
		);

#ifdef __CUDACC__
		__device__ frequency_map::Result d_compareFrequencyMaps_pip(
			frequency_map::FrequencyMap* input,
			FrequencyMapIndex_t other_index,
			frequency_map::FrequencyMap* output
		);
#endif
		__host__ frequency_map::Result h_compareFrequencyMaps_pip(
			frequency_map::FrequencyMap* input,
			FrequencyMapIndex_t other_index,
			frequency_map::FrequencyMap* output
		);
		__device__ __host__ frequency_map::FrequencyMap* getFrequencyMapPointer(
			FrequencyMapIndex_t other_index
		);
	};

	frequency_map::FrequencyMap* getFrequencyMapPointer(
		FrequencyMapIndex_t index
	);
}
