#pragma once
#include <memory>
#include "definitions.hpp"
#include "frequency_map.hpp"
#include <stdint.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <cstdio>

// Forward declaration to avoid circular dependency
namespace job {
	struct Job;
}

using std::cout;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;

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

		int32_t createInitialjobs(job::Job* buffer);

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

		__host__ void copyInputFrequencyMap(frequency_map::FrequencyMap* dest);

		__host__ void printSentence(
			shared_ptr<vector<FrequencyMapIndex_t>> indices
		);
	};

	frequency_map::FrequencyMap* getFrequencyMapPointer(
		FrequencyMapIndex_t index
	);
}
