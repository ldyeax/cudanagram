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
		throw new std::runtime_error("unspecified");
	}
	num_workers = 0;
	if (p_cpu) {
		worker::WorkerFactory* cpu_factory
			= worker::getWorkerFactory_CPU(database, dict);
		if (cpu_factory == nullptr) {
			cerr << "Anagrammer::initWorkers: CPU factory is null" << endl;
			throw new std::runtime_error("unspecified");
		}
		num_cpu_workers = cpu_factory->Spawn(
			workers,
			num_jobs_per_batch,
			database,
			dict
		);
		num_workers += num_cpu_workers;
		usleep(1000);
		fprintf(stderr, "Spawned %ld CPU workers\n", num_cpu_workers);
	}
	if (p_gpu) {
		worker::WorkerFactory* gpu_factory
			= worker::getWorkerFactory_GPU(database, dict);
		if (gpu_factory == nullptr) {
			cerr << "Anagrammer::initWorkers: GPU factory is null" << endl;
			throw new std::runtime_error("unspecified");
		}
		num_gpu_workers = gpu_factory->Spawn(
			workers + num_workers,
			num_jobs_per_batch - num_workers,
			database,
			dict
		);
		num_workers += num_gpu_workers;
		usleep(1000);
		fprintf(stderr, "Spawned %ld GPU workers\n", num_gpu_workers);
	}
	if (num_workers <= 0) {
		cerr << "Anagrammer::initWorkers: no workers spawned" << endl;
		throw new std::runtime_error("unspecified");
	}
	num_available_threads = 0;
	for (int64_t i = 0; i < num_workers; i++) {
		num_available_threads += workers[i]->numThreads();
	}

	std::int64_t max_int64 = std::numeric_limits<std::int64_t>::max();
	int64_t num_threads_in_one_cpu_worker = workers[0]->numThreads();
	int64_t jobIDs_per_thread = max_int64 / (num_available_threads + num_threads_in_one_cpu_worker * 2L);
	int64_t current_start = num_threads_in_one_cpu_worker * jobIDs_per_thread;

	for (int64_t i = 0; i < num_workers; i++) {
		workers[i]->setJobIDIncrementStart(current_start);
		cerr << "Worker " << i << ": start = " << current_start << ", ";
		current_start += workers[i]->numThreads() * jobIDs_per_thread;
		cerr << " end = " << current_start << endl;
		if (current_start < 0) {
			cerr << "Anagrammer::initWorkers: current_start overflowed" << endl;
			throw new std::runtime_error("unspecified");
		}
		if (max_int64 - current_start < workers[i]->numThreads() * jobIDs_per_thread) {
			cerr << "Anagrammer::initWorkers: current_start will overflow on next increment" << endl;
			throw new std::runtime_error("unspecified");
		}
	}

	fprintf(stderr, "Anagrammer::initWorkers: total available threads = %ld\n", num_available_threads);
	spawned_workers = true;

	insertStartJobs();
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
	throw new std::runtime_error("unspecified");
	Job startJob = {};
    dict->copyInputFrequencyMap(&startJob.frequency_map);
    startJob.start = 0;
    database->writeJob(startJob);
}

void Anagrammer::insertStartJobs()
{
	shared_ptr<vector<Job>> initial_jobs
		= dict->createInitialjobs(num_jobs_per_batch * 2);
	cerr << "Trying to create " << num_jobs_per_batch * 2 << " initial jobs, got "
		 << initial_jobs->size() << " jobs" << endl;
	vector<Job> tmp_finished = {};
	tmp_finished.reserve(initial_jobs->size());
	vector<Job> tmp_unfinished = {};
	tmp_unfinished.reserve(initial_jobs->size());
	for (int64_t i = 0; i < initial_jobs->size(); i++) {
		if (initial_jobs->at(i).finished) {
			tmp_finished.push_back(initial_jobs->at(i));
		}
		else {
			if (initial_jobs->at(i).frequency_map.isAllZero()) {
				cerr << "Initial job " << i << " has all-zero frequency map" << endl;
				throw new std::runtime_error("unspecified");
			}
			if (initial_jobs->at(i).frequency_map.anyNegative()) {
				cerr << "Initial job " << i << " has negative frequency map value" << endl;
				throw new std::runtime_error("unspecified");
			}
			tmp_unfinished.push_back(initial_jobs->at(i));
		}
	}
	int64_t total_threads_available = 0;
	for (int64_t i = 0; i < num_workers; i++) {
		total_threads_available += workers[i]->numThreads();
	}

	// Initial start increment values have already been set in initWorkers

	int64_t num_to_insert = tmp_unfinished.size();
	cerr << "Inserting " << tmp_finished.size() << " finished initial jobs and "
		 << num_to_insert << " unfinished initial jobs into database" << endl;

	if (tmp_finished.size() <= 0) {
		cerr << "No finished initial jobs to insert" << endl;
		throw new std::runtime_error("unspecified");
	}
	database->insertJobsWithIDs(tmp_finished.data(), tmp_finished.size());

	if (tmp_unfinished.size() <= 0) {
		cerr << "No unfinished initial jobs to insert" << endl;
		return ;
	}


	int64_t num_taken = 0;
	while (num_taken < tmp_unfinished.size()) {
		bool taken_this_iteration = false;
		for (int64_t i = 0; i < num_workers; i++) {
			auto &worker = workers[i];
			int64_t max_to_take = tmp_unfinished.size() - num_taken;
			if (max_to_take <= 0) {
				break;
			}
			int64_t taken = worker->takeJobsAndWrite(
				tmp_unfinished.data() + num_taken,
				max_to_take
			);
			printf("Gave %d initial jobs to worker %ld, it took %ld jobs\n",
				(int32_t)max_to_take, i, taken);
			if (taken > 0) {
				num_taken += taken;
				taken_this_iteration = true;
			}
		}
	}
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

void Anagrammer::waitForWorkersToFinish()
{
	bool all_finished = false;
	bool all_cpu_finished = false;
	bool all_gpu_finished = (num_workers <= num_cpu_workers);
	// cerr << "all_finished=" << all_finished
	// 	 << " all_cpu_finished=" << all_cpu_finished
	// 	 << " all_gpu_finished=" << all_gpu_finished << " (true if num_workers <= num_cpu_workers i.e. " << num_workers << " < " << num_cpu_workers << ")" << endl;
	while (!all_finished) {
		// cout << "Enter to continue while (!all_finished)" << endl;
		// std::cin >> dummy;
		bool tmp_all_cpu_finished = true;
		bool tmp_all_gpu_finished = num_workers <= num_cpu_workers;
		bool any_unfinished = false;
		int64_t total_jobs_taken_tmp = 0;
		for (int64_t i = 0; i < num_workers; i++) {
			if (workers[i]->async_jobs_taken <= 0) {
				#ifdef CUDANAGRAM_TESTING
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker " << workers[i]->id << " had no jobs taken, skipping finished check." << endl;
				}
				#endif
				continue;
			}
			total_jobs_taken_tmp += workers[i]->async_jobs_taken.load();
			//if (i < 4 || i >= num_cpu_workers) cerr << i << "=" << workers[i]->finished.load() << " ";
			if (!workers[i]->finished) {
				if (i < num_cpu_workers) {
					tmp_all_cpu_finished = false;
				}
				else {
					tmp_all_gpu_finished = false;
				}
				any_unfinished = true;
				//cerr << "Worker " << workers[i]->id << " not finished yet. "<< endl;
			}
		}
		if (!any_unfinished) {
			all_finished = true;
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "No unfinished workers found" << endl;
			}
			if (total_jobs_taken_tmp <= 0) {
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "No jobs were taken by any worker" << endl;
				}
				return;
			}
			return;
		}
		//cerr << endl;
		if (tmp_all_cpu_finished && !all_cpu_finished) {
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Finished all CPU workers" << endl;
			}
			all_cpu_finished = true;
		}
		if (tmp_all_gpu_finished && !all_gpu_finished) {
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Finished all GPU workers" << endl;
			}
			all_gpu_finished = true;
		}
		all_finished = all_cpu_finished && all_gpu_finished;
		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		//std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

void Anagrammer::run()
{
	if (!done_init) {
		cerr << "Anagrammer::run called before Anagrammer::init" << endl;
		throw new std::runtime_error("unspecified");
	}
	if (!spawned_workers) {
		cerr << "Anagrammer::run called before Anagrammer::initWorkers" << endl;
		throw new std::runtime_error("unspecified");
	}

	iteration = 0;
	string dummy;
	while (true) {
		// std::cin >> dummy;
		// cout << "Enter to start Anagrammer iteration " << (iteration + 1) << endl;
		iteration++;
		{
			std::lock_guard<std::mutex> lock(global_print_mutex);
			cerr << "Anagrammer iteration " << iteration << endl;
		}

		// {
		// 	std::lock_guard<std::mutex> lock(global_print_mutex);
		// 	num_unfinished_jobs = database->printJobsStats();
		// }

		// if (num_unfinished_jobs <= 0) {
		// 	{
		// 		std::lock_guard<std::mutex> lock(global_print_mutex);
		// 		cerr << "No unfinished jobs remaining, done." << endl;
		// 	}
		// 	break;
		// }
		// {
		// 	std::lock_guard<std::mutex> lock(global_print_mutex);
		// 	cerr << "Unfinished jobs remaining: " << num_unfinished_jobs << endl;
		// }

		for (int64_t i = 0; i < num_workers; i++) {
			workers[i]->reset();
		}

		{
			std::lock_guard<std::mutex> lock(global_print_mutex);
			cerr << "Reset all workers (num_workers = " << num_workers << ")" << endl;
		}

		int64_t taken_jobs = 0;

		for (int64_t i = 0; i < num_workers; i++) {
			while (!workers[i]->ready_to_take_jobs.load());
			workers[i]->async_jobs_taken.store(0);
			workers[i]->finished_taking_jobs.store(false);
			workers[i]->async_jobs_to_take.store(workers[i]->numThreads());
			workers[i]->ready_to_take_jobs.store(true);
		}
		for (int64_t i = 0; i < num_workers; i++) {
			//cerr << "Waiting for worker " << i << " to finish taking jobs" << endl;
			while (!workers[i]->finished_taking_jobs.load());
			taken_jobs += workers[i]->async_jobs_taken.load();;
			if (taken_jobs == 0) {
				continue;
			}
			// fprintf(stderr,
			// 	" Worker %ld took %ld jobs, total taken now %ld/%ld\n",
			// 	i, workers[i]->async_jobs_taken.load(), taken_jobs, num_unfinished_jobs
			// );
		}

		{
			std::lock_guard<std::mutex> lock(global_print_mutex);
			fprintf(stderr, " Took %ld jobs\n", taken_jobs);
		}
		// get current time
		auto start_time = std::chrono::high_resolution_clock::now();
		//for (int32_t i = 0; i < workers_assigned; i++) {
		for (int32_t i = 0; i < num_workers; i++) {
			//fprintf(stderr, " Starting worker %d\n", i);
			// if (workers[i]->async_jobs_taken <= 0) {
			// 	//cerr << "Worker " << i << " has no jobs taken, skipping doJobs_async()" << endl;
			// 	continue;
			// }
			workers[i]->doJobs_async();
		}
		//fprintf(stderr, " Started worker %d\n", i);
		// wait for all workers to finish
		waitForWorkersToFinish();
	exit_all_finished_loop:
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
		{
			std::lock_guard<std::mutex> lock(global_print_mutex);
			fprintf(stderr, " Jobs/second processed+write: %.2f\n", (num_unfinished_jobs / (duration / 1000.0)));
		}
		//database->finishJobs(unfinished_jobs, num_unfinished_jobs);
		for (int64_t i = 0; i < num_workers; i++) {
			workers[i]->reset();
		}
		{
			std::lock_guard<std::mutex> lock(global_print_mutex);
			fprintf(stderr, "Finished iteration %ld\n", iteration);
		}
		#ifdef DEBUG_WORKER_CPU
		if (database->has_found_sentence) {
			if (database->getSentenceJobCountSlow() <= 0) {
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Database indicates found sentence, but no sentence jobs found!" << endl;
				}
				goto judgement_day;
			}
		}
		#endif
	}
judgement_day:
	for (int64_t i = 0; i < num_workers; i++) {
		workers[i]->terminate();
		while (!workers[i]->judgement_day.load()) {
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Waiting for worker " << i << " to terminate..." << endl;
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
	{
		std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Terminated all workers" << endl;
	}
}

void Anagrammer::printFoundSentences()
{
	database->printFoundSentences(dict);
}

void Anagrammer::printDict()
{
	dict->printDict();
}
