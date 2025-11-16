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
    };
    class Worker {
	public:
		std::atomic<int32_t> id;

		std::atomic<bool> terminated{false};
		std::atomic<bool> judgement_day{false};
		virtual void terminate();

		std::atomic<bool> ready_to_start{false};
		std::atomic<int64_t> async_jobs_to_take = 0;
		std::atomic<int64_t> async_jobs_taken = 0;
		std::atomic<bool> ready_to_take_jobs{false};
		std::atomic<bool> finished_taking_jobs{false};
		virtual void doJobs();
		virtual void loop() = 0;
		virtual void doJob(job::Job* input, int64_t count) = 0;
		std::atomic<vector<Job>*> unfinished_jobs = new vector<Job>();

        std::atomic<Result*> last_result = new Result();
        Worker(database::Database* db, dictionary::Dictionary* dict);
		std::atomic<database::Database*> thread_db = nullptr;
		Database* main_thread_db = nullptr;
        std::atomic<dictionary::Dictionary*> dict = nullptr;
        std::atomic<bool> finished{false};
		void reset();
        /**
         * Takes up to max_length jobs from the buffer, returns number of jobs taken.
		 * Call multiple times to keep giving jobs for the next round -
		 *  if a worker "prefers" only 1 job, but you have more jobs than active workers,
		 *  the thing to do is to keep looping over the workers and giving them jobs
		 *  before starting the processing batch
         **/
        virtual int64_t takeJobs(Job* buffer, int64_t max_length);
        /**
         * Takes up to max_length jobs from the buffer,
		 *  writes the jobs to its own database,
		 *  and returns number of jobs taken.
		 * Call multiple times to keep giving jobs for the next round -
		 *  if a worker "prefers" only 1 job, but you have more jobs than active workers,
		 *  the thing to do is to keep looping over the workers and giving them jobs
		 *  before starting the processing batch
         **/
        virtual int64_t takeJobsAndWrite(Job* buffer, int64_t max_length);
		/**
		 * Take up to max_length jobs from own child database, returns number of jobs taken
		 * Call multiple times to keep giving jobs for the next round -
		 * if a worker "prefers" only 1 job, but you have more jobs than active workers,
		 * the thing to do is to keep looping over the workers and giving them jobs
		 * before starting the processing batch
		 **/
		virtual int64_t takeJobs(int64_t max_length);
        virtual void doJobs_async();
        virtual int32_t numThreads() = 0;
        void WriteResult(Result* result, Dictionary* dict, database::Txn* txn);
		void setJobIDIncrementStart(int64_t start)
		{
			thread_db.load()->setJobIDIncrementStart(start);
		}
		void finishJobs();
		void finishJobs(database::Txn* txn);
		void finishJobs_async();
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
