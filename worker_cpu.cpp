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

class Worker_CPU : public Worker {
private:
	int32_t id;

public:
	void reset() override
	{
		Worker::reset();
		unfinished_jobs.clear();
	}
	std::thread thread;
	Worker_CPU(database::Database* p_db, dictionary::Dictionary* p_dict, int32_t p_id)
		: Worker(p_db, p_dict)
	{
		id = p_id;
		//fprintf(stderr, "Constructed CPU Worker\n");
		thread = std::thread(&Worker_CPU::loop, this);
		thread.detach();
	}

	int64_t takeJobs(Job* buffer, int64_t max_length) override
	{
		finished = false;
		if (max_length <= 0) {
			throw;
		}
		int64_t jobs_to_take = std::min((int64_t)numThreads(), max_length);
		for (int64_t i = 0; i < jobs_to_take; i++) {
			unfinished_jobs.push_back(buffer + i);
		}
		return jobs_to_take;
	}


	void doJobs_async() override
	{
		finished = false;
		ready_to_start = true;
	}

	int32_t numThreads() override
	{
		return 16;  // CPU cores are ~4-5x faster than CUDA cores at sequential work
	}

	void doJob(job::Job* p_input, int64_t p_count) override
	{
		frequency_map::FrequencyMap tmp = {};
		job::Job tmp_job = {};
		tmp_job.parent_job_id = p_input->job_id;
		FrequencyMapIndex_t start = p_input->start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			throw;
		}

		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict->h_compareFrequencyMaps_pip(
				&p_input->frequency_map,
				i,
				&tmp_job.frequency_map
			);
			if (result == INCOMPLETE_MATCH) {
				tmp_job.start = i;
				tmp_job.is_sentence = false;
				tmp_job.finished = false;
				last_result.new_jobs.push_back(tmp_job);
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				tmp_job.is_sentence = true;
				tmp_job.finished = true;
				last_result.new_jobs.push_back(tmp_job);
			}
		}
	}


};

class WorkerFactory_CPU : public WorkerFactory {
public:
    int64_t Spawn(
        Worker** buffer,
        int64_t max,
        database::Database* db,
        dictionary::Dictionary* dict
    ) override {
		unsigned int concurrent_threads = std::thread::hardware_concurrency();

		if (concurrent_threads > 0) {
			std::cerr << "The system supports approximately " << concurrent_threads << " CPU threads." << std::endl;
		} else {
			std::cerr << "The number of CPU threads could not be determined or is zero." << std::endl;
			concurrent_threads = 16;
		}
		if (concurrent_threads <= 16) {
			concurrent_threads--;
		}
		else if (concurrent_threads <= 32) {
			concurrent_threads -= 2;
		}
		else {
			concurrent_threads -= 4;
		}
		std::cerr << "Spawning up to " << concurrent_threads << " CPU workers." << std::endl;
        int64_t count = 0;
        for (int64_t i = 0; i < max && i < concurrent_threads; i++) {
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

