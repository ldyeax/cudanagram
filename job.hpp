#pragma once
#define JobID_t int64_t
#include <cstdint>
#include <stdint.h>
#include "definitions.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include <iostream>
#include <cstdio>
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
		bool finished;
		bool is_sentence;

		void print()
		{
			fprintf(stderr, "Job %ld: parent_job_id=%ld frequency_map=", job_id, parent_job_id);
			frequency_map.print();
			fprintf(stderr, " start=%d finished=%d is_sentence=%d\n", start, finished, is_sentence);
		}

		__device__ void d_print()
		{
			printf("Job %ld: parent_job_id=%ld frequency_map=", job_id, parent_job_id);
			frequency_map.d_print();
			printf(" start=%d finished=%d is_sentence=%d\n", start, finished, is_sentence);
		}
	};
}
