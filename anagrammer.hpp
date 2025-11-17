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
        Dictionary* dict;
        Database* database;
        string input;

		Worker** workers;
		int64_t num_workers;

		shared_ptr<vector<Job>> initial_jobs;

		void spawnWorkers(
			bool use_cpu,
			bool use_gpu
		)
		{
			workers = new Worker*[1024];
			num_workers = 0;
			int64_t max_int64 = 9223372036854775807L;
			int64_t total_threads = 0;
			vector<WorkerFactory*> factories = {};
			if (use_cpu) {
				factories.push_back(
					worker::getWorkerFactory_CPU(
						database, dict
					)
				);
			}
			if (use_gpu) {
				factories.push_back(
					worker::getWorkerFactory_GPU(
						database, dict
					)
				);
			}
			for (auto& f : factories) {
				total_threads += f->getTotalThreads();
			}
			// Allocation: JobID_t 1 through x for initial jobs
			// Rest is allocated to the remaining space of int64_t
			//  based on total threads
			auto initial_jobs = dict->createInitialJobs(
				total_threads
			);

			if (initial_jobs.unfinished_jobs->size() == 0) {
				cerr << "No initial unfinished jobs created" << endl;
				return;
			}

			database->insertJobsWithIDs(
				initial_jobs.non_sentence_finished_jobs->data(),
				initial_jobs.non_sentence_finished_jobs->size()
			);
			database->insertJobsWithIDs(
				initial_jobs.unfinished_jobs->data(),
				initial_jobs.unfinished_jobs->size()
			);

			int64_t total_jobs
				= initial_jobs.unfinished_jobs->size()
				+ initial_jobs.non_sentence_finished_jobs->size();

			int64_t min_job_id = initial_jobs.max_id * 2;
			int64_t ids_to_allocate = max_int64 - min_job_id;
			int64_t ids_per_thread = ids_to_allocate / total_threads;
			int64_t jobs_per_thread = total_jobs / total_threads;
			Job* initial_jobs_buffer = initial_jobs.unfinished_jobs->data();
			int64_t num_jobs_taken = 0;
			for (auto& f : factories) {
				int64_t ids_for_factory =
					f->getTotalThreads() * ids_per_thread;

				int64_t num_jobs_for_factory
					= f->getTotalThreads() * jobs_per_thread;
				if (num_jobs_taken + num_jobs_for_factory > total_jobs) {
					num_jobs_for_factory = total_jobs - num_jobs_taken;
				}
				num_jobs_taken += num_jobs_for_factory;

				num_workers += f->spawn(
					workers + num_workers,
					dict,
					min_job_id,
					min_job_id + ids_for_factory,
					initial_jobs_buffer,
					num_jobs_for_factory
				);
				min_job_id += ids_for_factory;
				std::cout << (max_int64 - min_job_id) << std::endl;
				initial_jobs_buffer += num_jobs_for_factory;
			}

			assert(num_jobs_taken == total_jobs);
			assert(initial_jobs_buffer == initial_jobs.unfinished_jobs->data() + total_jobs);

			cout << "Waiting for all workers to initialize..";
			bool all_initialized = false;
			while (!all_initialized) {
				for (int64_t i = 0; i < num_workers; i++) {
					if (workers[i]->worker_status == worker::uninitialized) {
						goto still_uninitialized;
					}
				}
				all_initialized = true;
still_uninitialized:
				cerr << " ..";
				continue;
			}
			cerr << endl;
			cout << "All workers initialized" << endl;
		}

    public:
		Anagrammer(
			string p_input,
			bool use_cpu = true,
			bool use_gpu = true,
			string dictionary_filename = "dictionary.txt"
		)
		{
			if (!use_cpu && !use_gpu) {
				throw;
			}
			dict = new Dictionary(
				(char*)p_input.c_str(),
				(char*)dictionary_filename.c_str(),
				nullptr,
				0
			);
			input = p_input;
			spawnWorkers(use_cpu, use_gpu);
		}
	};
}
