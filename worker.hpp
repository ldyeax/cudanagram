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

using job::Job;
using database::Database;
using database::Txn;
using std::vector;
using dictionary::Dictionary;
using std::atomic;

namespace worker {
	enum WorkerStatus {
		uninitialized,
		spawning,
		running,
		ended
	};
	class Worker {
	private:
		int64_t start_id;
		/**
		 * Our own buffer copy of unfinished jobs to process,
		 *  either copied from the initial seeding
		 *  or fetched from our own database in later iterations
		 **/
		Job* unfinished_jobs;
		int64_t num_unfinished_jobs;
	public:
		virtual int64_t getBufferSize()
		{
			return 65535L;
		}
		volatile WorkerStatus worker_status = uninitialized;
		char* volatile database_name;
		void innerLoop()
		{
			while (num_unfinished_jobs > 0) {
				getJobsFromDatabase();
				if (num_unfinished_jobs <= 0) {
					break;
				}
				doJobs();
			}
		}
		Worker(
			/**
			 * Jobs will be created starting with this ID,
			 *  though the initial jobs given may have (
			 *   almost certainly will have) lower IDs
			 **/
			int64_t p_start_id,

			Job* p_initial_jobs,
			int64_t p_num_initial_jobs
		) 
		{
			start_id = p_start_id;
			unfinished_jobs = new Job[getBufferSize()];
			num_unfinished_jobs = p_num_initial_jobs;
		}

	};
	class WorkerFactory {
	public:
		/**
		 * Spawns up to max workers, returns number of workers spawned
		 **/
		virtual int64_t Spawn(
			Worker** buffer,
			int64_t max,
			Database* db,
			Dictionary* dict
		) = 0;
	};
	extern WorkerFactory* getWorkerFactory_CPU(database::Database* db, dictionary::Dictionary* dict);
	extern WorkerFactory* getWorkerFactory_GPU(database::Database* db, dictionary::Dictionary* dict);
}
