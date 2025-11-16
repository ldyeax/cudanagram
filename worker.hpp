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

namespace worker {
	class Worker {
	public:
		std::atomic<int32_t> id;

		Worker(Job* p_start_jobs, int64_t p_n_start_jobs) {

		}


		Job* volatile start_jobs;
		std::atomic<int64_t> num_start_jobs;

		void loop()
		{

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
