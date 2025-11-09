#include <cstdint>
#include <stdint.h>
#include "definitions.hpp"
#include "dictionary.hpp"
#define JobID_t int64_t
namespace job {
	struct Job {
		/**
		 * >= 1
		 * **/
		JobID_t job_id;
		/**
		 * 0 for no parent
		 * **/
		JobID_t parent_job_id;
		frequency_map::FrequencyMap frequency_map;
		FrequencyMapIndex_t start;
		/**
		 * Job sets this when spawning a new job.
		 * parent_frequency_map_index of the first jobs will all be -1,
		 *  as they started from nothing
		 * **/
		FrequencyMapIndex_t parent_frequency_map_index;
		bool finished;
	};
}
