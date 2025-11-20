#pragma once
#include "definitions.hpp"
#include <memory>
#include "job.hpp"
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
using job::Job;



#define FrequencyMapIndex_t int32_t

namespace dictionary {
	struct dictstats {
	    int32_t initial_words_parsed = 0;
	    int32_t initial_words_removed = 0;
		int32_t frequency_map_rejections = 0;
	};

	struct InitialJobsCreation {
		shared_ptr<vector<Job>> unfinished_jobs = nullptr;
		shared_ptr<vector<Job>> non_sentence_finished_jobs = nullptr;
		shared_ptr<vector<Job>> sentence_finished_jobs = nullptr;
		JobID_t max_id = 0;
	};

	class Dictionary {
	private:
		void init();
	public:
		FILE* output_file = stdout;
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

		//shared_ptr<vector<Job>> createInitialjobs(int64_t count);
		InitialJobsCreation createInitialJobs(
			int64_t count,
			int8_t max_depth
		);

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
			FILE* output_file,
			shared_ptr<vector<FrequencyMapIndex_t>> indices
		);

		void printWordsAt(FrequencyMapIndex_t fm_index);
		void printWordsAt(FrequencyMapIndex_t fm_index, int32_t depth);
		void printDict();
	};

	frequency_map::FrequencyMap* getFrequencyMapPointer(
		FrequencyMapIndex_t index
	);
}
