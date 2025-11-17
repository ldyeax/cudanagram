#pragma once

#include <memory>
#include "job.hpp"
#include "frequency_map.hpp"
#include "database.hpp"
#include "dictionary.hpp"
#include <stdint.h>
#include <vector>
#include <atomic>
#include <thread>
#include <cstring>

using job::Job;
using database::Database;
using database::Txn;
using std::vector;
using dictionary::Dictionary;
using std::atomic;

namespace worker {
	enum WorkerStatus {
		uninitialized,
		running,
		ended
	};
	/**
	 * Worker is expected to be constructed in a new thread -
	 *  it will enter its loop dirctly from the constructor.
	 **/
	class Worker {
	protected:
		int64_t start_id;
		/**
		 * Our own buffer copy of unfinished jobs to process,
		 *  either copied from the initial seeding
		 *  or fetched from our own database in later iterations
		 **/
		Job* unfinished_jobs;
		int64_t num_unfinished_jobs;
		Database* database;
		Dictionary* dictionary;
		virtual int64_t getBufferSize()
		{
			return 65535L;
		}
		/**
		 * Child class is expected to implement init,
		 *  which will be called right before the loop starts
		 **/
		virtual void init() = 0;
		void loop()
		{
			init();
			worker_status = running;
			while (num_unfinished_jobs > 0) {
				getJobsFromDatabase();
				if (num_unfinished_jobs <= 0) {
					break;
				}
				doJobs();
			}
			worker_status = ended;
		}
	public:
		volatile WorkerStatus worker_status = uninitialized;
		volatile bool failed = false;
		/**
		 * Worker will write to this in its init function
		 **/
		char* volatile database_name = nullptr;

		Worker(
			/**
			 * Jobs will be created starting with this ID,
			 *  though the initial jobs given may have (
			 *   almost certainly will have) lower IDs
			 **/
			int64_t p_start_id,
			/**
			 * Buffer to copy initial jobs from
			 **/
			Job* p_initial_jobs,
			/**
			 * Number of jobs that will be copied from buffer
			 **/
			int64_t p_num_initial_jobs
		)
		{
			try {
				if (p_initial_jobs == nullptr) {
					throw;
				}
				start_id = p_start_id;
				unfinished_jobs = new Job[getBufferSize()];
				num_unfinished_jobs = p_num_initial_jobs;
				memcpy(
					unfinished_jobs,
					p_initial_jobs,
					sizeof(Job) * p_num_initial_jobs
				);
				loop();
			}
			catch (...) {
				failed = true;
				// todo: throw full exception
			}
		}

	};
	class WorkerFactory {
	public:
		virtual int64_t getNumWorkers() = 0;
		/**
		 * CPU worker factory would return number of system threads,
		 * GPU would return number of CUDA threads it can use across
		 *  all available GPUs
		 **/
		virtual int64_t getTotalThreads() = 0;
		/**
		 * Returns number of workers spawned
		 **/
		virtual int64_t spawn(
			Worker** buffer,
			Dictionary* dict,
			int64_t min_new_job_id,
			int64_t max_new_job_id,
			/**
			 * May have finished jobs duplicated
			 **/
			Job* initial_jobs,
			int64_t num_initial_jobs
		) = 0;
	};
	extern WorkerFactory* getWorkerFactory_CPU(database::Database* db, dictionary::Dictionary* dict);
	extern WorkerFactory* getWorkerFactory_GPU(database::Database* db, dictionary::Dictionary* dict);
}
