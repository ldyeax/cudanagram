#pragma once

#include "definitions.hpp"
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include "worker.hpp"
#include <stdint.h>
#include <string>
#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
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
        string input;

		atomic<atomic<Worker*>*> volatile workers;
		atomic<int64_t> num_workers;

		shared_ptr<vector<Job>> initial_jobs;

		void spawnWorkers(
			bool use_cpu,
			bool use_gpu
		)
		{
			workers = new atomic<Worker*>[1024];
			num_workers = 0;
			int64_t total_threads = 0;
			vector<WorkerFactory*> factories = {};
			if (use_cpu) {
				factories.push_back(
					worker::getWorkerFactory_CPU()
				);
			}
			if (use_gpu) {
				factories.push_back(
					worker::getWorkerFactory_GPU()
				);
			}
			for (auto& f : factories) {
				total_threads += f->getTotalThreads();
			}

			auto initial_jobs = dict->createInitialJobs(
				total_threads
			);

			if (initial_jobs.unfinished_jobs->size() == 0) {
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "No initial unfinished jobs created" << endl;
				}
				return;
			}

			#ifdef CUDANAGRAM_TESTING
			for (auto& unfinished_job : *initial_jobs.unfinished_jobs) {
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					// cerr << "Initial unfinished job: ";
					// unfinished_job.print();
					if (unfinished_job.finished) {
						cerr << "ERROR: initial unfinished job is marked finished!" << endl;
						throw;
					}
					if (unfinished_job.is_sentence) {
						cerr << "ERROR: initial unfinished job is marked is_sentence!" << endl;
						throw;
					}
					if (unfinished_job.frequency_map.isAllZero()) {
						cerr << "ERROR: initial unfinished job has all-zero frequency map!" << endl;
						throw;
					}
				}
			}
			#endif

			int64_t total_jobs = initial_jobs.unfinished_jobs->size();

			int64_t jobs_per_thread = total_jobs / total_threads;
			Job* initial_jobs_buffer = initial_jobs.unfinished_jobs->data();
			int64_t num_jobs_taken = 0;
			for (auto& f : factories) {
				int64_t num_jobs_for_factory
					= f->getTotalThreads() * jobs_per_thread;
				if (num_jobs_taken + num_jobs_for_factory != total_jobs) {
					num_jobs_for_factory = total_jobs - num_jobs_taken;
				}
				num_jobs_taken += num_jobs_for_factory;

				// Get the pointer from the atomic and offset it
				atomic<Worker*>* workers_ptr = workers.load();
				num_workers += f->spawn(
					workers_ptr + num_workers,
					dict,
					initial_jobs_buffer,
					num_jobs_for_factory,
					initial_jobs.non_sentence_finished_jobs
				);
				initial_jobs_buffer += num_jobs_for_factory;
			}

			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Spawned " << num_workers
					<< " workers for total of "
					<< total_jobs << "(" << num_jobs_taken << ") jobs"
					<< endl;
			}

			assert(num_jobs_taken == total_jobs);
			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Initial jobs vector buffer end at "
					<< (void*)(initial_jobs.unfinished_jobs->data() + total_jobs)
					<< ", current pointer at "
					<< (void*)initial_jobs_buffer
					<< endl;
			}
			assert(initial_jobs_buffer == initial_jobs.unfinished_jobs->data() + total_jobs);

			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Waiting for all workers to initialize..";
			}
			bool all_initialized = false;
			while (!all_initialized) {
				for (int64_t i = 0; i < num_workers; i++) {
					if (workers.load()[i].load() == nullptr) {
						{
							//std::lock_guard<std::mutex> lock(global_print_mutex);
							cerr << "Worker " << i << " at " << &(workers.load()[i]) << " is null.." << endl;
						}
						goto still_uninitialized;
					}
					if (workers[i].load()->worker_status == worker::uninitialized) {
						goto still_uninitialized;
					}
				}
				all_initialized = true;
still_uninitialized:
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					// cerr << " .." << endl;
				}
				std::this_thread::sleep_for(std::chrono::seconds(1));
				continue;
			}
			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "All workers initialized" << endl;
			}
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

			bool all_finished = true;
			do {
				all_finished = true;
				for (int64_t i = 0; i < num_workers; i++) {
					if (workers[i] == nullptr) {
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						all_finished = false;
						cerr << "Worker.2 " << i << " is null.." << endl;
						break;
					}
					else if (workers[i].load()->worker_status != worker::ended) {
						all_finished = false;
						break;
					}
				}
				if (!all_finished) {
					std::this_thread::sleep_for(std::chrono::seconds(1));
				}
			} while (!all_finished);

			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "All workers finished" << endl;
			}
		}
	};
}
