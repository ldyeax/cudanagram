#include <cstdint>
#include <stdint.h>
#include "definitions.hpp"
#include "dictionary.hpp"
#define JobID_t int64_t
namespace job {
	struct Job {
		JobID_t job_id;
		JobID_t parent_job_id;
		dictionary::frequency_map::FrequencyMap frequency_map;
		FrequencyMapIndex_t start;
	};
}
