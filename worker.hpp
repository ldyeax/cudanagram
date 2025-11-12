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
    struct Result {
        vector<Job> new_jobs;
        // vector<Job> found_sentences;
    };
    class Worker {
	public:
		std::atomic<bool> ready_to_start{false};
		virtual void doJobs();
		virtual void loop();
		virtual void doJob(job::Job input) = 0;
		vector<Job*> unfinished_jobs;

        Result last_result = {};
        Worker(database::Database* db, dictionary::Dictionary* dict);
        database::Database* db = nullptr;
        dictionary::Dictionary* dict = nullptr;
        std::atomic<bool> finished{false};
		virtual void reset()
		{
			last_result.new_jobs.clear();
			//last_result.found_sentences.clear();
		}
        /**
         * Takes up to max_length jobs from the buffer, returns number of jobs taken.
		 * Call multiple times to keep giving jobs for the next round -
		 *  if a worker "prefers" only 1 job, but you have more jobs than active workers,
		 *  the thing to do is to keep looping over the workers and giving them jobs
		 *  before starting the processing batch
         **/
        virtual int64_t takeJobs(Job* buffer, int64_t max_length) = 0;
        virtual void doJobs_async();
        virtual int32_t numThreads() = 0;
        void WriteResult(Result* result, dictionary::Dictionary* dict, database::Txn* txn);
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
