#define NUM_JOBS_PER_BATCH (1024*512)

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


int main()
{
	std::signal(SIGINT, handle_sigint);

    Dictionary* dict = new Dictionary(
		"twomilkmengocomedy",
		"dictionary.txt",
		NULL,
		-1
	);
	dict->printStats();
    Database* database = new Database();
    printf("Constructed database %p\n", database);
    // Worker* worker = worker::workerFactory_CPU(database, dict);
    // printf("Constructed CPU worker %p\n", worker);
    Job startJob = {};
    dict->copyInputFrequencyMap(&startJob.frequency_map);
    startJob.start = 0;
    database->writeUnfinishedJob(startJob);
    int64_t iteration = 0;

    worker::WorkerFactory* factory = worker::getWorkerFactory_CPU(database, dict);
    worker::Worker* workers[NUM_JOBS_PER_BATCH];
    int32_t num_workers = factory->Spawn(
        &workers[0],
        NUM_JOBS_PER_BATCH,
        database,
        dict
    );
    printf("Spawned %d CPU workers\n", num_workers);
    Job* unfinished_jobs = new Job[NUM_JOBS_PER_BATCH];

    int32_t num_available_threads = 0;
    for (int32_t i = 0; i < num_workers; i++) {
        num_available_threads += workers[i]->numThreads();
    }

    while (true) {
        Txn* txn = database->beginTransaction();
        int32_t num_unfinished_jobs
            = database->getUnfinishedJobs(NUM_JOBS_PER_BATCH, unfinished_jobs, txn);
        if (num_unfinished_jobs <= 0) {
            break;
        }
        int32_t taken_jobs = 0;
        // If there are 16 workers and 1024 jobs, wach worker gets 1024/16 jobs
		int32_t workers_assigned = 0;
		bool first_assignment_iteration = true;
        while (taken_jobs < num_unfinished_jobs) {
			bool taken_this_loop = false;
			for (int32_t i = 0; i < num_workers; i++) {
				int32_t max_jobs_to_take = num_unfinished_jobs - taken_jobs;
				if (max_jobs_to_take <= 0) {
					break;
				}
				taken_jobs += workers[i]->takeJobs(
					unfinished_jobs + taken_jobs,
					max_jobs_to_take
				);
				taken_this_loop = true;
				if (first_assignment_iteration) {
					workers_assigned++;
				}
			}
			first_assignment_iteration = false;
			if (!taken_this_loop) {
				cerr << " No jobs taken in loop " << endl;
				return 1;
			}
			//printf(" Taken %d/%d jobs so far\n", taken_jobs, num_unfinished_jobs);
		}
		printf(" Took %d jobs\n", taken_jobs);
		for (int32_t i = 0; i < workers_assigned; i++) {
			//printf(" Starting worker %d\n", i);
			workers[i]->doJobs_async();
			//printf(" Started worker %d\n", i);
		}
		for (int32_t i = 0; i < workers_assigned; i++) {
			while (!workers[i]->finished) {
			//	printf(" Waiting for worker %d to finish...\n", i);
				usleep(100000);
			}
			//printf(" Writing results for worker %d\n", i);
			workers[i]->WriteResult(nullptr, nullptr, txn);
			// cout << " Worker " << i << " finished" << endl;
		}
		iteration++;
		database->finishJobs(unfinished_jobs, num_unfinished_jobs, txn);
		database->commitTransaction(txn);
		printf("Finished iteration %ld\n", iteration);
		// if (iteration >= 3) {
		// 	return 0;
		// }
    }
    return 0;
}
