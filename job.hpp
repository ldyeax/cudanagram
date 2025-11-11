#pragma once
#include <cstdint>
#include <stdint.h>
#include "definitions.hpp"
#include "dictionary.hpp"
#include <iostream>
#define JobID_t int64_t
using std::endl;
using std::cout;
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
		// /**
		//  * Job sets this when spawning a new job.
		//  * parent_frequency_map_index of the first jobs will all be -1,
		//  *  as they started from nothing
		//  * **/
		// FrequencyMapIndex_t parent_frequency_map_index;
		bool finished;

		constexpr void const print()
		{
			//std::cout << "Job " << job_id << ": parent_job_id=" << parent_job_id << " frequency_map=";
			printf("Job %ld: parent_job_id=%ld frequency_map=", job_id, parent_job_id);
			frequency_map.print();
			//std::cout << " start=" << start << " parent_frequency_mapIndex=" << parent_frequency_map_index << " finished=" << finished << endl;
			//printf(" start=%d parent_frequency_map_index=%d finished=%d\n", start, parent_frequency_map_index, finished);
			printf(" start=%d finished=%d\n", start, finished);
		}
	};
}
