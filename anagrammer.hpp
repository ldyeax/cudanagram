#pragma once

#include "definitions.hpp"
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include "worker.hpp"
#include <stdint.h>
#include <string>
#include <cstdio>
using dictionary::Dictionary;
using job::Job;
using std::shared_ptr;
using std::vector;
using database::Database;
using std::string;
using worker::Worker;
using worker::WorkerFactory;
namespace anagrammer {
    class Anagrammer {
	private:

    public:
		int64_t num_jobs_per_batch;

		int64_t num_gpu_workers = 0;
		int64_t num_cpu_workers = 0;

        Dictionary* dict;
        Database* database;
        string input;
		Worker** workers;
		vector<WorkerFactory> worker_factories;

        Anagrammer(int64_t p_num_jobs_per_batch, string p_input);
        Anagrammer(int64_t p_num_jobs_per_batch, string p_input, Dictionary* p_dict);
		Anagrammer(int64_t p_num_jobs_per_batch, string p_input, string p_dict_filename);

    };
}
