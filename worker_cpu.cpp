#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include <thread>
#include <iostream>

using namespace worker;

using job::Job;
using std::cout;
using std::endl;

/**
    struct Result {
        vector<Job> new_jobs;
        vector<Job> found_sentences;
    };
 **/

class Worker_CPU : public Worker {
private:
	int32_t id;
	std::atomic<bool> ready_to_start{false};
	void loop()
	{
		while (true) {
			while (!ready_to_start) {
				std::this_thread::yield();
			}
			ready_to_start = false;
			finished = false;
			//cout << "Worker_CPU " << id << " starting doJobs()" << endl;
			doJobs();
			//cout << "Worker_CPU " << id << " finished doJobs(), creating " << last_result.new_jobs.size() << " new jobs and " << last_result.found_sentences.size() << " found sentences." << endl;
			finished = true;
		}
	}
	vector<Job*> unfinished_jobs;
public:
	std::thread thread;
	Worker_CPU(database::Database* p_db, dictionary::Dictionary* p_dict, int32_t p_id)
		: Worker(p_db, p_dict)
	{
		id = p_id;
		//printf("Constructed CPU Worker\n");
		thread = std::thread(&Worker_CPU::loop, this);
		thread.detach();
	}

	int32_t takeJobs(Job* buffer, int32_t max_length) override
	{
		finished = false;
		if (max_length <= 0) {
			throw;
		}
		unfinished_jobs.push_back(buffer);
		return 1;
	}

	void doJobs()
	{
		finished = false;
		last_result.new_jobs = vector<Job>{unfinished_jobs.size()};
		last_result.found_sentences = vector<Job>{unfinished_jobs.size()};
		for (int32_t i = 0; i < unfinished_jobs.size(); i++) {
			doJob(*unfinished_jobs[i]);
		}
		finished = true;
		//cout << "Worker_CPU finished " << num_unfinished_jobs << " jobs and set the atomic bool to true." << endl;
	}

	void doJobs_async() override
	{
		finished = false;
		ready_to_start = true;
	}

	int32_t numThreads() override
	{
		return 1;
	}

	void doJob(job::Job input)
	{
		frequency_map::FrequencyMap tmp = {};
		job::Job tmp_job = {};
		tmp_job.parent_job_id = input.job_id;
		FrequencyMapIndex_t start = input.start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			throw;
		}

		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict->h_compareFrequencyMaps_pip(
				&input.frequency_map,
				i,
				&tmp_job.frequency_map
			);
			if (result == INCOMPLETE_MATCH) {
				tmp_job.start = i;
				last_result.new_jobs.push_back(tmp_job);
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				last_result.found_sentences.push_back(tmp_job);
			}
		}
	}


};

class WorkerFactory_CPU : public WorkerFactory {
public:
    int32_t Spawn(
        Worker** buffer,
        int32_t max,
        database::Database* db,
        dictionary::Dictionary* dict
    ) override {
		unsigned int concurrent_threads = std::thread::hardware_concurrency();

		if (concurrent_threads > 0) {
			std::cout << "The system supports approximately " << concurrent_threads << " CPU threads." << std::endl;
		} else {
			std::cout << "The number of CPU threads could not be determined or is zero." << std::endl;
			concurrent_threads = 16;
		}
        int32_t count = 0;
        for (int32_t i = 0; i < max && i < concurrent_threads; i++) {
            buffer[i] = new Worker_CPU(db, dict, i);
            count++;
        }
        return count;
    }
};

WorkerFactory* worker::getWorkerFactory_CPU(database::Database* db, dictionary::Dictionary* dict)
{
	static WorkerFactory* ret = new WorkerFactory_CPU();
	return ret;
}

