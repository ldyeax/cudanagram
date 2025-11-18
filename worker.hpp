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
		Job* unfinished_jobs = nullptr;
		int64_t num_unfinished_jobs = 0;
		Job* new_jobs_buffer = nullptr;
		int64_t num_new_jobs = 0;
		Database* database = nullptr;
		Dictionary* dictionary = nullptr;
		virtual int64_t getUnfinishedJobsBufferSize()
		{
			//return 16384L;
			return 65535L;
		}
		int64_t getNewJobsBufferSize()
		{
			if (dictionary == nullptr) {
				cerr << "dictionary is null in getNewJobsBufferSize" << endl;
				throw new std::runtime_error(
					"dictionary is null in getNewJobsBufferSize"
				);
			}
			return getUnfinishedJobsBufferSize() * (int64_t) dictionary->frequency_maps_length;
		}
		/**
		 * Child class is expected to implement init,
		 *  which will be called right before the loop starts
		 **/
		virtual void init() = 0;
		void getUnfinishedJobsFromDatabase()
		{
			num_unfinished_jobs = database->getUnfinishedJobs(
				getUnfinishedJobsBufferSize(),
				unfinished_jobs
			);
		}
		virtual void doJobs() = 0;
		void writeNewJobsToDatabase()
		{
			// cerr << "Base writeNewJobsToDatabase called with num_new_jobs = "
			// 	<< num_new_jobs << endl;
			if (num_new_jobs > 0) {
				database->writeNewJobs(
					new_jobs_buffer,
					num_new_jobs
				);
				num_new_jobs = 0;
			}
		}
		void loop()
		{
			static int id_ = 0;
			int id = ++id_;
			init();
			//cerr << "Worker initialized, entering main loop" << endl;
			worker_status = running;
			getUnfinishedJobsFromDatabase();
			string output_file_name = "sqlite/_worker_" + std::to_string(id) + "_output.txt";
			FILE* output_file = fopen(output_file_name.c_str(), "w");
			if (output_file == nullptr) {
				cerr << "Worker " << id << " failed to open output file " << output_file_name << endl;
				throw std::runtime_error("Failed to open worker output file");
			}
			int64_t _count = 0;
			while (num_unfinished_jobs > 0) {
				_count++;
				// cerr << "Worker " << id << " starting doJobs with "
				// 	<< num_unfinished_jobs << " unfinished jobs at generation " << _count << endl;

				doJobs();
				writeNewJobsToDatabase();
				{
					// //std::lock_guard<std::mutex> lock(global_print_mutex);
					if (_count % 1024 == id * 4) {
						database->printFoundSentences(dictionary, output_file);
						fflush(output_file);
					}
				}
				getUnfinishedJobsFromDatabase();
			}
			database->printFoundSentences(dictionary, output_file);
			fclose(output_file);
			cerr << "Worker " << id << " finished all jobs, exiting loop" << endl;
			worker_status = ended;
		}
	public:
		atomic<WorkerStatus> worker_status = uninitialized;
		atomic<bool> failed = false;
		/**
		 * Worker will write to this in its init function
		 **/
		atomic<char*> database_name = nullptr;
		void start() {
			cerr << "Worker starting loop()" << endl;
			loop();
		}
		Worker(
			Dictionary* dict,
			/**
			 * Buffer to copy initial jobs from
			 **/
			Job* p_initial_jobs,
			/**
			 * Number of jobs that will be copied from buffer
			 **/
			int64_t p_num_initial_jobs,
			shared_ptr<vector<Job>> non_sentence_finished_jobs
		)
		{
			try {
				if (p_initial_jobs == nullptr) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Initial jobs is null in Worker constructor" << endl;
					}
					throw;
				}
				if (dict == nullptr) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Dictionary is null in Worker constructor" << endl;
					}
					throw;
				}
				if (p_num_initial_jobs <= 0) {
					{
						//std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Initial jobs count " << p_num_initial_jobs << " is invalid in base Worker constructor" << endl;
					}
					throw;
				}
				// if (p_initial_jobs == nullptr) {
				// else {
				// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
				// 	cerr << "Initial jobs is valid in Worker constructor" << endl;
				// }

				database = new Database();
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					database_name = (char*)database->db_name.c_str();
					//cerr << "Worker database name: " << database_name << endl;
				}
				database->setJobIDIncrementStart(0x7FFFFFFF);
				dictionary = dict;
				unfinished_jobs = new Job[getUnfinishedJobsBufferSize()];
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					// cerr << "Allocated unfinished jobs buffer of size "
					// 	<< getUnfinishedJobsBufferSize() << endl;
				}
				// num_unfinished_jobs = p_num_initial_jobs;
				// memcpy(
				// 	unfinished_jobs,
				// 	p_initial_jobs,
				// 	sizeof(Job) * p_num_initial_jobs
				// );
				// {
				// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
				// 	cerr << "Copied initial jobs into worker buffer" << endl;
				// }
				//cerr << p_num_initial_jobs << " initial jobs: " << endl;
				#ifdef DEBUG_WORKER_CPU
				for (int64_t i = 0; i < p_num_initial_jobs; i++) {
					cerr << "Initial job " << i << ": ";
					p_initial_jobs[i].print();
					if (p_initial_jobs[i].frequency_map.isAllZero()) {
						cerr << "ERROR: initial job has all-zero frequency map!" << endl;
						throw;
					}
				}
				#endif
				database->writeNewJobs(
					p_initial_jobs,
					p_num_initial_jobs
				);
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr <<
					 "Inserted initial jobs into worker database" << endl;
				}
				#endif
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Allocating new jobs buffer of size "
						<< getNewJobsBufferSize() << endl;
				}
				#ifdef DEBUG_WORKER_CPU
				#endif
				new_jobs_buffer = new Job[getNewJobsBufferSize()];
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Allocated new jobs buffer of size "
						<< getNewJobsBufferSize() << endl;
					fprintf(stderr, "non_sentence_finished_jobs = %p\n", (void*)non_sentence_finished_jobs.get());
				}
				#endif
				if (non_sentence_finished_jobs->size() > 0) {
					database->insertJobsWithIDs(
						non_sentence_finished_jobs->data(),
						non_sentence_finished_jobs->size()
					);
				}
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Inserted non-sentence finished jobs into worker database" << endl;
				}
				for (auto& job : *non_sentence_finished_jobs) {
					{
						std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Non-sentence finished job inserted: ";
						job.print();
						database->getJob(job.job_id).print();
					}
				}
				#endif
			}
			// catch all exception types
			catch (std::exception& e) {
				failed = true;
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker caught exception in constructor: " << e.what() << std::endl;
				}
				throw e;
			}
			catch (...) {
				failed = true;
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker caught unknown exception in constructor" << std::endl;
				}
				std::rethrow_exception(std::current_exception());
			}
		}

	};
	class WorkerFactory {
	public:
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
			atomic<Worker*>* buffer,
			Dictionary* dict,
			/**
			 * May have finished jobs duplicated
			 **/
			Job* initial_jobs,
			int64_t num_initial_jobs,
			shared_ptr<vector<Job>> non_sentence_finished_jobs
		) = 0;
	};
	extern WorkerFactory* getWorkerFactory_CPU();
	extern WorkerFactory* getWorkerFactory_GPU();
}
