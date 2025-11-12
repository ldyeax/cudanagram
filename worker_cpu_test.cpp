// #define NUM_JOBS_PER_BATCH (1024LL*1024LL*1024LL*2LL)

#include <memory>
#include "worker.hpp"
#include "database.hpp"
#include "dictionary.hpp"
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include "anagrammer.hpp"

using std::vector;
using database::Database;
using worker::Worker;
using dictionary::Dictionary;
using job::Job;
using database::Txn;
using std::cout;
using std::cerr;
using std::endl;

void handle_sigint(int signal)
{
    std::cerr << "\nCaught Ctrl-C (signal " << signal << "), exiting\n";
    std::quick_exit(1);
}

int main(int argc, char* argv[])
{
	std::signal(SIGINT, handle_sigint);
	string dummy;

	anagrammer::Anagrammer a(
		(1024LL*1024LL),
		"twomilkmengocomedy"
	);
	a.initWorkers(true, false);
	a.run();

	return 0;
}


// int main_1()
// {
// 	std::signal(SIGINT, handle_sigint);
// 	string dummy;

//     Dictionary* dict = new Dictionary(
// 		//"mementomoriiftheninethlionatethesun",
// 		"twomilkmengocomedy",
// 		"dictionary.txt",
// 		NULL,
// 		-1
// 	);
// 	dict->printStats();
//     Database* database = new Database();
//     fprintf(stderr, "Constructed database %p\n", database);
//     // Worker* worker = worker::workerFactory_CPU(database, dict);
//     // fprintf(stderr, "Constructed CPU worker %p\n", worker);

// 	Job startJob = {};
//     dict->copyInputFrequencyMap(&startJob.frequency_map);
//     startJob.start = 0;
//     database->writeUnfinishedJob(startJob);


//     int64_t iteration = 0;

//     worker::WorkerFactory* factory = worker::getWorkerFactory_CPU(database, dict);
// 	worker::Worker** workers = new Worker*[NUM_JOBS_PER_BATCH];
//     int64_t num_workers = factory->Spawn(
//         &workers[0],
//         NUM_JOBS_PER_BATCH,
//         database,
//         dict
//     );
//     fprintf(stderr, "Spawned %ld CPU workers\n", num_workers);
//     Job* unfinished_jobs = new Job[NUM_JOBS_PER_BATCH];

// 	// int32_t num_initial = dict->createInitialjobs(unfinished_jobs);
// 	// fprintf(stderr, "Created %d initial jobs\n", num_initial);
// 	// database->writeJobs(unfinished_jobs, num_initial);

//     int64_t num_available_threads = 0;
//     for (int64_t i = 0; i < num_workers; i++) {
//         num_available_threads += workers[i]->numThreads();
//     }

//     while (true) {
// 		auto start_time_whole_loop = std::chrono::high_resolution_clock::now();
//         Txn* txn = database->beginTransaction();
//         int64_t num_unfinished_jobs
//             = database->getUnfinishedJobs(NUM_JOBS_PER_BATCH, unfinished_jobs, txn);
// 		database->commitTransaction(txn);
// 		database->printJobsStats();

//         if (num_unfinished_jobs <= 0) {
//             break;
//         }

// 		for (int64_t i = 0; i < num_workers; i++) {
// 			workers[i]->reset();
// 		}

//         int64_t taken_jobs = 0;
//         // If there are 16 workers and 1024 jobs, each worker gets 1024/16 jobs
// 		int64_t workers_assigned = 0;
// 		bool first_assignment_iteration = true;
//         while (taken_jobs < num_unfinished_jobs) {
// 			//fprintf(stderr, " Assigning jobs: taken %ld/%ld\n", taken_jobs, num_unfinished_jobs);
// 			bool taken_this_loop = false;
// 			for (int64_t i = 0; i < num_workers; i++) {
// 				//fprintf(stderr, "i=%ld ", i);
// 				int64_t max_jobs_to_take = num_unfinished_jobs - taken_jobs;
// 				if (max_jobs_to_take <= 0) {
// 					//fprintf(stderr, " All jobs taken\n");
// 					break;
// 				}
// 				taken_jobs += workers[i]->takeJobs(
// 					unfinished_jobs + taken_jobs,
// 					max_jobs_to_take
// 				);
// 				//fprintf(stderr, " Worker %ld took jobs, total taken now %ld/%ld\n", i, taken_jobs, num_unfinished_jobs);
// 				taken_this_loop = true;
// 				if (first_assignment_iteration) {
// 					workers_assigned++;
// 				}
// 			}
// 			first_assignment_iteration = false;
// 			if (!taken_this_loop) {
// 				cerr << " No jobs taken in loop " << endl;
// 				return 1;
// 			}
// 			//fprintf(stderr, " Taken %ld/%ld jobs so far\n", taken_jobs, num_unfinished_jobs);
// 		}
// 		fprintf(stderr, " Took %ld jobs\n", taken_jobs);
// 		// get current time
// 		auto start_time = std::chrono::high_resolution_clock::now();
// 		for (int32_t i = 0; i < workers_assigned; i++) {
// 			//fprintf(stderr, " Starting worker %d\n", i);
// 			workers[i]->doJobs_async();
// 			//fprintf(stderr, " Started worker %d\n", i);
// 		}
// 		bool all_finished = false;
// 		string to_finish_string = "UPDATE job SET finished = TRUE WHERE job_id = ANY($1::BIGINT[])";
// 		while (!all_finished) {
// 			for (int32_t i = 0; i < workers_assigned; i++) {
// 				if (!workers[i]->finished) {
// 					all_finished = false;
// 					break;
// 				}
// 				all_finished = true;
// 			}
// 		}
// 		auto end_time = std::chrono::high_resolution_clock::now();
// 		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
// 		fprintf(stderr, " Jobs/second processed+write: %.2f\n", (num_unfinished_jobs / (duration / 1000.0)));
// 		// auto start_write_results_time = std::chrono::high_resolution_clock::now();
// 		// for (int32_t i = 0; i < workers_assigned; i++) {
// 		// 	// while (!workers[i]->finished) {
// 		// 	// //	fprintf(stderr, " Waiting for worker %d to finish...\n", i);
// 		// 	// 	usleep(100);
// 		// 	// }
// 		// 	//fprintf(stderr, " Writing results for worker %d\n", i);
// 		// 	workers[i]->WriteResult(nullptr, nullptr, txn);
// 		// 	// cerr << " Worker " << i << " finished" << endl;
// 		// }
// 		// auto end_write_results_time = std::chrono::high_resolution_clock::now();
// 		// auto duration_write_results = std::chrono::duration_cast<std::chrono::milliseconds>(end_write_results_time - start_write_results_time).count();
// 		// fprintf(stderr, " WriteResult() time: %ld ms\n", duration_write_results);
// 		// fprintf(stderr, " WriteResult() Jobs/second: %.2f\n", (num_unfinished_jobs / (duration_write_results / 1000.0)));
// 		iteration++;
// 		// auto start_finishjobs_time = std::chrono::high_resolution_clock::now();
// 		// database->finishJobs(unfinished_jobs, num_unfinished_jobs, txn);
// 		// auto end_finishjobs_time = std::chrono::high_resolution_clock::now();
// 		// auto duration_finishjobs = std::chrono::duration_cast<std::chrono::milliseconds>(end_finishjobs_time - start_finishjobs_time).count();
// 		// fprintf(stderr, " finishJobs() time: %ld ms\n", duration_finishjobs);
// 		// fprintf(stderr, " finishJobs() Jobs/second: %.2f\n", (num_unfinished_jobs / (duration_finishjobs / 1000.0)));
// 		// auto start_commit_time = std::chrono::high_resolution_clock::now();
// 		// database->commitTransaction(txn);
// 		// auto end_commit_time = std::chrono::high_resolution_clock::now();
// 		// auto duration_commit = std::chrono::duration_cast<std::chrono::milliseconds>(end_commit_time - start_commit_time).count();
// 		// fprintf(stderr, " commitTransaction() time: %ld ms\n", duration_commit);
// 		// fprintf(stderr, " commitTransaction() Jobs/second: %.2f\n", (num_unfinished_jobs / (duration_commit / 1000.0)));
// 		auto end_time_whole_loop = std::chrono::high_resolution_clock::now();
// 		auto duration_whole_loop = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_whole_loop - start_time_whole_loop).count();
// 		fprintf(stderr, " Whole loop time: %ld ms\n", duration_whole_loop);
// 		fprintf(stderr, " Whole loop Jobs/second: %.2f\n", (num_unfinished_jobs / (duration_whole_loop / 1000.0)));
// 		fprintf(stderr, "Finished iteration %ld\n", iteration);
// 		// if (iteration >= 3) {
// 		// 	return 0;
// 		// }
// 		//std::getline(std::cin, dummy);
//     }

// 	database->printFoundSentences(dict);
//     return 0;
// }
