#include <memory>
#include "worker.hpp"
#include "database.hpp"
#include "dictionary.hpp"
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <vector>

using std::vector;
using database::Database;
using worker::Worker;
using dictionary::Dictionary;
using job::Job;

int main()
{    
    Dictionary* dict = new Dictionary(
		"twomilkmengocomedy",
		"dictionary.txt",
		NULL,
		-1
	);
	dict->printStats();
    Database* database = new Database();
    printf("Constructed database %p\n", database);
    Worker* worker = worker::workerFactory_CPU(database, dict);
    printf("Constructed CPU worker %p\n", worker);
    Job startJob = {};
    dict->copyInputFrequencyMap(&startJob.frequency_map);
    startJob.start = 0;
    database->writeJob(startJob);
    int64_t iteration = 0;
    while (true) {
        Job* unfinished_jobs;
        int32_t num_unfinished_jobs = database->getUnfinishedJobs(1024, unfinished_jobs);
        // todo: another call to getUnfinishedJobs in the meantime would still return the same jobs
        printf("Got %d unfinished jobs at %p\n", num_unfinished_jobs, unfinished_jobs);
        if (num_unfinished_jobs <= 0) {
            break;
        }
        vector<worker::Result> results;
        for (int32_t i = 0; i < num_unfinished_jobs; i++) {
            auto result = worker->doJob(unfinished_jobs[i]);
            results.push_back(result);
        }
        for (auto& result : results) {
            worker->WriteResult(result, dict);
        }
        database->finishJobs(unfinished_jobs, num_unfinished_jobs);
        printf("Finished iteration %ld\n", iteration++);
    }
    return 0;
}