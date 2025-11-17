#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include <thread>
#include <iostream>
#include <mutex>

using namespace worker;

using job::Job;
using std::cout;
using std::cerr;
using std::endl;

class Worker_CPU : public Worker {
private:
public:
	Worker_CPU(
		dictionary::Dictionary* p_dict,
		Job* p_initial_jobs,
		int64_t p_num_initial_jobs,
		shared_ptr<vector<Job>> non_sentence_finished_jobs
	) : Worker(
		p_dict,
		p_initial_jobs,
		p_num_initial_jobs,
		non_sentence_finished_jobs
	)
	{
		{
			//std::lock_guard<std::mutex> lock(global_print_mutex);
			fprintf(
				stderr,
				"Started CPU Worker with initial jobs at %p\n",
				p_initial_jobs
			);
		}
	}

	virtual void init() override
	{

	}

	void doJob(Job& job)
	{
		for (FrequencyMapIndex_t i = job.start; i < dictionary->frequency_maps_length; i++) {
			job::Job tmp = {};
			#ifdef DEBUG_WORKER_CPU
			cerr << "Worker_CPU doing job "
				<< job.job_id << " comparing with frequency map index " << i << endl;
			#endif
			frequency_map::FrequencyMap* dict_fm
				= dictionary->getFrequencyMapPointer(i);
			frequency_map::Result res
				= dictionary->h_compareFrequencyMaps_pip(
					&job.frequency_map,
					i,
					&tmp.frequency_map
				);
			Job tmp2;
			switch (res) {
			case COMPLETE_MATCH:
			#ifdef DEBUG_WORKER_CPU
				cerr << "Found complete match for job "
					<< job.job_id << " at frequency map index " << i << endl;
				dict_fm->print();
				tmp2 = job;
				do {
					tmp2.print();
					tmp2.frequency_map.print();
					if (tmp2.parent_job_id != 0) {
						tmp2 = database->getJob(tmp2.parent_job_id);
					}
				} while (tmp2.parent_job_id != 0);
			#endif
				tmp.finished = true;
				tmp.is_sentence = true;
			case INCOMPLETE_MATCH:
				tmp.parent_job_id = job.job_id;
				tmp.start = i;
				new_jobs_buffer[num_new_jobs++] = tmp;
			default:
				break;
			}
		}
	}

	virtual void doJobs() override
	{
		num_new_jobs = 0;
		// Simple example: mark all unfinished jobs as finished
		for (int64_t i = 0; i < num_unfinished_jobs; i++) {
			Job& job = unfinished_jobs[i];
			doJob(job);
		}
	}
};

class WorkerFactory_CPU : public WorkerFactory {
public:
	int64_t getTotalThreads() override
	{
		//return 1;

		unsigned int num_threads
			= std::thread::hardware_concurrency();
		if (num_threads < 2) {
			throw;
		}
		return num_threads - 1;
	}
	int64_t spawn(
		/**
		 * Worker buffer to write spawned workers into
		 **/
		atomic<Worker*>* buffer,
		/**
		 * Dictionary to use for workers
		 **/
		Dictionary* dict,
		/**
		 * Initial unfinished jobs to copy from
		 **/
		Job* initial_jobs,
		/**
		 * Number of initial unfinished jobs to copy from
		 **/
		int64_t num_initial_jobs,
		shared_ptr<vector<Job>> non_sentence_finished_jobs
	) override
	{
		int64_t num_threads = getTotalThreads();
		{
			//std::lock_guard<std::mutex> lock(global_print_mutex);
			cerr << "CPU Worker Factory spawning "
				<< num_threads << " threads"
				<< " for " << num_initial_jobs << " initial jobs"
				<< endl;
		}
		int64_t threads_per_worker = num_initial_jobs / num_threads;
		if (threads_per_worker < 1) {
			threads_per_worker = 1;
		}
		int64_t jobs_given = 0;
		for (unsigned int i = 0; i < num_threads; i++) {
			int64_t jobs_to_give = threads_per_worker;
			//cerr << "Jobs to give: " << jobs_to_give << endl;
			if (jobs_given + jobs_to_give > num_initial_jobs) {
				jobs_to_give = num_initial_jobs - jobs_given;
				//cerr << "Adjusted jobs to give: " << jobs_to_give << endl;
			}
			if (jobs_to_give <= 0) {
				cerr << "No more jobs to give, breaking spawn loop" << endl;
				break;
			}
			Job* thread_initial_jobs = initial_jobs;
			// create as detached thread
			std::thread t1([i, jobs_to_give, thread_initial_jobs, dict, non_sentence_finished_jobs, buffer]() {
				cerr << "Thread got jobs to give: " << jobs_to_give << endl;
				if (jobs_to_give <= 0) {
					cerr << "This should be impossible " << jobs_to_give << endl;
					throw;
				}
				buffer[i].store(new Worker_CPU(
					dict,
					thread_initial_jobs,
					jobs_to_give,
					non_sentence_finished_jobs
				));
				{
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					fprintf(
						stderr,
						"CPU Worker %d created at %p ;;;;;;;\n",
						i,
						buffer[i].load()
					);
				}
				buffer[i].load()->start();

				{
					// sleep
					//std::this_thread::sleep_for(std::chrono::seconds(1));
					//std::lock_guard<std::mutex> lock(global_print_mutex);
					fprintf(
						stderr,
						"Worker %d set buffer at %p\n",
						i,
						buffer[i].load()
					);
				}
			});
			// std::this_thread::sleep_for(std::chrono::seconds(1));
			// {
			// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
			// 	cerr << "Created thread for worker " << i << endl;
			// }
			t1.detach();
			jobs_given += jobs_to_give;
			initial_jobs += jobs_to_give;
		}
		return num_threads;
	}
};

WorkerFactory* worker::getWorkerFactory_CPU()
{
	static WorkerFactory* ret = new WorkerFactory_CPU();
	return ret;
}
