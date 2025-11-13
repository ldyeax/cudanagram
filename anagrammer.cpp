//#define NUM_JOBS_PER_BATCH 1024*512

// #define TEST_ANAGRAMMER 1

#include "definitions.hpp"
#include "anagrammer.hpp"
#include <iostream>
#include <unistd.h>
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include <stdint.h>
#include <string>
#include "definitions.hpp"
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include "worker.hpp"
#include <stdint.h>
#include <string>
using dictionary::Dictionary;
using job::Job;
using std::shared_ptr;
using std::vector;
using database::Database;
using std::cout;
using std::endl;
using std::string;
using dictionary::Dictionary;
using job::Job;
using std::shared_ptr;
using std::vector;
using database::Database;
using std::string;
using worker::Worker;
using worker::WorkerFactory;

using namespace anagrammer;

using std::cerr;
using std::endl;

void Anagrammer::initWorkers(bool p_cpu, bool p_gpu)
{
	if (!done_init) {
		cerr << "Anagrammer::initWorkers called before Anagrammer::init" << endl;
		throw;
	}
	num_workers = 0;
	if (p_cpu) {
		worker::WorkerFactory* cpu_factory
			= worker::getWorkerFactory_CPU(database, dict);
		if (cpu_factory == nullptr) {
			cerr << "Anagrammer::initWorkers: CPU factory is null" << endl;
			throw;
		}
		num_cpu_workers = cpu_factory->Spawn(
			workers,
			num_jobs_per_batch,
			database,
			dict
		);
		num_workers += num_cpu_workers;
		fprintf(stderr, "Spawned %ld CPU workers\n", num_cpu_workers);
	}
	if (p_gpu) {
		worker::WorkerFactory* gpu_factory
			= worker::getWorkerFactory_GPU(database, dict);
		if (gpu_factory == nullptr) {
			cerr << "Anagrammer::initWorkers: GPU factory is null" << endl;
			throw;
		}
		num_gpu_workers = gpu_factory->Spawn(
			workers + num_workers,
			num_jobs_per_batch - num_workers,
			database,
			dict
		);
		num_workers += num_gpu_workers;
		fprintf(stderr, "Spawned %ld GPU workers\n", num_gpu_workers);
	}
	if (num_workers <= 0) {
		cerr << "Anagrammer::initWorkers: no workers spawned" << endl;
		throw;
	}
	num_available_threads = 0;
	for (int64_t i = 0; i < num_workers; i++) {
		num_available_threads += workers[i]->numThreads();
	}
	fprintf(stderr, "Anagrammer::initWorkers: total available threads = %ld\n", num_available_threads);
	spawned_workers = true;
}

/**
 * Called after assigning input and constructing the dictionary
 */
void Anagrammer::init()
{
	dict->printStats();
    database = new Database();
    unfinished_jobs = new Job[num_jobs_per_batch];
	workers = new Worker*[num_jobs_per_batch];
	done_init = true;
	fprintf(stderr, "Initialized Anagrammer with input %s\n", input.data());
}

void Anagrammer::insertStartJob()
{
	Job startJob = {};
    dict->copyInputFrequencyMap(&startJob.frequency_map);
    startJob.start = 0;
    database->writeUnfinishedJob(startJob);
}

Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input)
{
	num_jobs_per_batch = p_num_jobs_per_batch;
    input = p_input;
    dict = new Dictionary(
		p_input.data(),
		"dictionary.txt",
		NULL,
		-1
	);
	init();
}

Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input, Dictionary* p_dict)
{
    num_jobs_per_batch = p_num_jobs_per_batch;
    input = p_input;
    dict = p_dict;
	init();
}

Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input, string p_dict_filename)
{
    num_jobs_per_batch = p_num_jobs_per_batch;
    input = p_input;
    dict = new Dictionary(
		p_input.data(),
		p_dict_filename.data(),
		NULL,
		-1
	);
	init();
}

void Anagrammer::run()
{
	if (!done_init) {
		cerr << "Anagrammer::run called before Anagrammer::init" << endl;
		throw;
	}
	if (!spawned_workers) {
		cerr << "Anagrammer::run called before Anagrammer::initWorkers" << endl;
		throw;
	}
	insertStartJob();

	iteration = 0;

	while (true) {
		iteration++;
		cerr << "Anagrammer iteration " << iteration << endl;

		database->printJobsStats();

		cerr << "Getting unfinished jobs .." << endl;

		database::Txn* txn = database->beginTransaction();
		int64_t num_unfinished_jobs
			= database->getUnfinishedJobs(num_jobs_per_batch, unfinished_jobs, txn);
		database->commitTransaction(txn);
		database->printJobsStats();

		cerr << "Got unfinished jobs from database" << endl;

		if (num_unfinished_jobs <= 0) {
			cerr << "No unfinished jobs remaining, done." << endl;
			break;
		}
		cerr << "Unfinished jobs remaining: " << num_unfinished_jobs << endl;

		for (int64_t i = 0; i < num_workers; i++) {
			workers[i]->reset();
		}

		cerr << "Reset all workers" << endl;

		int64_t taken_jobs = 0;
		// If there are 16 workers and 1024 jobs, each worker gets 1024/16 jobs
		int64_t workers_assigned = 0;
		bool first_assignment_iteration = true;
		while (taken_jobs < num_unfinished_jobs) {
#ifdef TEST_ANAGRAMMER
			fprintf(stderr, " Assigning jobs: taken %ld/%ld\n", taken_jobs, num_unfinished_jobs);
#endif
			bool taken_this_loop = false;
			for (int64_t i = 0; i < num_workers; i++) {
#ifdef TEST_ANAGRAMMER
				fprintf(stderr, "i=%ld ", i);
#endif
				int64_t max_jobs_to_take = num_unfinished_jobs - taken_jobs;
				if (max_jobs_to_take <= 0) {
#ifdef TEST_ANAGRAMMER
					fprintf(stderr, " All jobs taken\n");
#endif
					break;
				}
				taken_jobs += workers[i]->takeJobs(
					unfinished_jobs + taken_jobs,
					max_jobs_to_take
				);
				//fprintf(stderr, " Worker %ld took jobs, total taken now %ld/%ld\n", i, taken_jobs, num_unfinished_jobs);
				taken_this_loop = true;
				if (first_assignment_iteration) {
					workers_assigned++;
				}
			}
			first_assignment_iteration = false;
			if (!taken_this_loop) {
				cerr << " No jobs taken in loop " << endl;
				return;
			}
#if defined(TEST_ANAGRAMMER)
			fprintf(stderr, " Taken %ld/%ld jobs so far\n", taken_jobs, num_unfinished_jobs);
#endif
		}
		fprintf(stderr, " Took %ld jobs\n", taken_jobs);
		// get current time
		auto start_time = std::chrono::high_resolution_clock::now();
		for (int32_t i = 0; i < workers_assigned; i++) {
			//fprintf(stderr, " Starting worker %d\n", i);
			workers[i]->doJobs_async();
		}
		//fprintf(stderr, " Started worker %d\n", i);
		// wait for all workers to finish
		bool all_finished = false;
		bool all_cpu_finished = false;
		bool all_gpu_finished = (workers_assigned <= num_cpu_workers);
		cerr << "all_finished=" << all_finished
			 << " all_cpu_finished=" << all_cpu_finished
			 << " all_gpu_finished=" << all_gpu_finished << " (true if workers_assigned <= num_cpu_workers i.e. " << workers_assigned << " < " << num_cpu_workers << ")" << endl;
		while (!all_finished) {
			bool tmp_all_cpu_finished = true;
			bool tmp_all_gpu_finished = workers_assigned <= num_cpu_workers;
			for (int64_t i = 0; i < workers_assigned; i++) {
				if (i < 4 || i >= num_cpu_workers) cerr << i << "=" << workers[i]->finished.load() << " ";
				if (!workers[i]->finished.load()) {
					if (i < num_cpu_workers) {
						tmp_all_cpu_finished = false;
					}
					else {
						tmp_all_gpu_finished = false;
					}
				}
			}
			cerr << endl;
			if (tmp_all_cpu_finished && !all_cpu_finished) {
				cerr << "Finished all CPU workers" << endl;
				all_cpu_finished = true;
			}
			if (tmp_all_gpu_finished && !all_gpu_finished) {
				cerr << "Finished all GPU workers" << endl;
				all_gpu_finished = true;
			}
			all_finished = all_cpu_finished && all_gpu_finished;
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		fprintf(stderr, " Jobs/second processed+write: %.2f\n", (num_unfinished_jobs / (duration / 1000.0)));
	}
}

void Anagrammer::printFoundSentences()
{
	database->printFoundSentences(dict);
}

