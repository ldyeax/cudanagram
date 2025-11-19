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
#include <algorithm> // For std::shuffle
#include <vector>    // For std::vector
#include <random>    // For random number generation
#include <chrono>    // For seeding with time
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

		atomic<Worker*> workers[1024];
		atomic<int64_t> num_workers;

		//shared_ptr<vector<Job>> initial_jobs;

		void spawnWorkers(
			bool use_cpu,
			bool use_gpu,
			bool resume
		)
		{
			//workers = new atomic<Worker*>[1024];
			for (int i = 0; i < 1024; i++) {
				workers[i].store(nullptr);
			}
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
			dictionary::InitialJobsCreation initial_jobs;
			if (!resume) {
				initial_jobs = dict->createInitialJobs(
					total_threads
				);

				if (initial_jobs.unfinished_jobs == nullptr) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "No initial unfinished jobs vector created" << endl;
					}
					return;
				}
				if (initial_jobs.unfinished_jobs->size() == 0) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "No initial unfinished jobs created" << endl;
					}
					return;
				}
				if (initial_jobs.non_sentence_finished_jobs == nullptr) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "No initial non-sentence finished jobs vector created" << endl;
					}
					return;
				}
				if (initial_jobs.sentence_finished_jobs == nullptr) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "No initial sentence finished jobs vector created" << endl;
					}
					return;
				}

				// Shuffle the initial unfinished jobs to distribute workload evenly
				{
					auto& jobs = *initial_jobs.unfinished_jobs;
					unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
					std::shuffle(jobs.begin(), jobs.end(), std::default_random_engine(seed));
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
				cerr << factories.size() << " factories" << endl;
				for (int64_t i = 0; i < (int64_t)factories.size(); i++) {
					auto& f = factories[i];
					int64_t num_jobs_for_factory
						= f->getNumJobsToGive();
					if (num_jobs_for_factory <= 0) {
						num_jobs_for_factory
							= f->getTotalThreads() * jobs_per_thread;
					}
					cerr << "num_jobs_for_factory.1 " << i << " = "
						<< num_jobs_for_factory << endl;

					if (i == (int64_t)factories.size() - 1) {
						// Last factory takes all remaining jobs
						num_jobs_for_factory = total_jobs - num_jobs_taken;
					}
					num_jobs_taken += num_jobs_for_factory;

					cerr << "num_jobs_for_factory.2 " << i << " = "
						<< num_jobs_for_factory << endl;

					//continue;

					// Get the pointer from the atomic and offset it
					//atomic<Worker*>* workers_ptr = workers.load();
					num_workers += f->spawn(
						//workers_ptr + num_workers,
						&workers[num_workers],
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
			}
			else {
				// List all .db files under ./sqlite/
				vector<Database*> existing_databases = database::getExistingDatabases();
				if (existing_databases.size() == 0) {
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "No existing databases found to resume from!" << endl;
					throw;
				}
				}
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Resuming from " << existing_databases.size() << " databases" << endl;
				}
				num_workers = 0;
				for (auto& f : factories) {
					int64_t db_to_give
						= existing_databases.size() - num_workers;
					if (db_to_give <= 0) {
						break;
					}
					num_workers += f->spawn(
						&workers[num_workers],
						dict,
						existing_databases.data() + num_workers,
						db_to_give
					);
				}

			}
			bool all_initialized = false;
			while (!all_initialized) {
				for (int64_t i = 0; i < num_workers; i++) {
					if (workers[i].load(std::memory_order_acquire) == nullptr) {
						{
							//std::lock_guard<std::mutex> lock(global_print_mutex);
							cerr << "Worker " << i << " at " << &workers[i] << " is null.." << endl;
						}
						goto still_uninitialized;
					}
					if (workers[i].load(std::memory_order_acquire)->worker_status == worker::uninitialized) {
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
			string dictionary_filename = "dictionary.txt",
			bool resume = false
		)
		{
			if (!use_cpu && !use_gpu) {
				throw;
			}
			if (resume) {
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Resuming from existing databases" << endl;
				}
			}
			dict = new Dictionary(
				(char*)p_input.c_str(),
				(char*)dictionary_filename.c_str(),
				nullptr,
				0
			);
			input = p_input;
			spawnWorkers(use_cpu, use_gpu, resume);

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
