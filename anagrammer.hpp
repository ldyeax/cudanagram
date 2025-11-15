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
		bool done_init = false;
		bool spawned_workers = false;
		void init();
		void insertStartJob();
		void insertStartJobs();
    public:
		int64_t num_jobs_per_batch;

		int64_t num_gpu_workers = 0;
		int64_t num_cpu_workers = 0;

        Dictionary* dict;
        Database* database;
        string input;
        Job* unfinished_jobs;

		int64_t num_available_threads;
		Worker** workers;
		int64_t num_workers;
		int64_t num_unfinished_jobs;
		int64_t iteration;
		vector<WorkerFactory> worker_factories;

        Anagrammer(int64_t p_num_jobs_per_batch, string p_input);
        Anagrammer(int64_t p_num_jobs_per_batch, string p_input, Dictionary* p_dict);
		Anagrammer(int64_t p_num_jobs_per_batch, string p_input, string p_dict_filename);
		void initWorkers(bool p_cpu, bool p_gpu);
        void run();
		void printFoundSentences();
		// Print the dictionary after rejection passes included frequency map
		void printDict();
    };
}
