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
	void printStatus()
	{
		std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Worker_CPU " << id
			<< " terminated=" << terminated
			<< " judgement_day=" << judgement_day
			<< " ready_to_start=" << ready_to_start
			<< " async_jobs_to_take=" << async_jobs_to_take
			<< " async_jobs_taken=" << async_jobs_taken
			<< " ready_to_take_jobs=" << ready_to_take_jobs
			<< " finished_taking_jobs=" << finished_taking_jobs
			<< " finished=" << finished
			<< endl;
	}
	void loop_internal()
	{
		std::mutex loop_mutex;
		while (!terminated) {
			#ifdef DEBUG_WORKER_CPU
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Worker_CPU " << id << " !terminated" << endl;
			}
			printStatus();
			#endif
			if (terminated) {
				judgement_day = true;
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " terminating at top of loop." << endl;
				}
				printStatus();
				#endif
				return;
			}
			// Wait for ready_to_start signal
			while (!ready_to_start) {
				if (terminated) {
					#ifdef DEBUG_WORKER_CPU
					{
						std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Worker_CPU " << id << " terminating during wait for ready_to_start." << endl;
					}
					printStatus();
					#endif
					judgement_day = true;
					return;
				}
				ready_to_take_jobs = true;
				std::this_thread::yield();
				if (terminated) {
					cerr << "Worker_CPU " << id << " terminating during wait for ready_to_start." << endl;
					judgement_day = true;
					return;
				}
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " waiting for ready_to_start... ";
				}
				printStatus();
				#endif
				if (async_jobs_to_take > 0) {
					#ifdef DEBUG_WORKER_CPU
					{
						std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Worker_CPU " << id << " taking " << async_jobs_to_take << " jobs." << endl;
					}
					printStatus();
					#endif


					ready_to_take_jobs = false;
					int64_t _taken = takeJobs(async_jobs_to_take);
					#ifdef DEBUG_WORKER_CPU
					{
						std::lock_guard<std::mutex> lock(global_print_mutex);
						cerr << "Worker_CPU " << id << " took " << _taken << " jobs." << endl;
					}
					printStatus();
					#endif
					async_jobs_taken.store(_taken);
					finished_taking_jobs = true;
				}
				#ifdef DEBUG_WORKER_CPU
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
				}
				#endif
			}
			unfinished_jobs.load()->clear();
			if (terminated) {
				judgement_day = true;
				cerr << "Worker_CPU " << id << " terminating after ready_to_start signal." << endl;
				printStatus();
				return;
			}
			#ifdef DEBUG_WORKER_CPU
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Worker_CPU " << id << " detected ready_to_start signal." << endl;
			}
			printStatus();
			#endif
			ready_to_start = false;
			#ifdef DEBUG_WORKER_CPU
			{
				std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Worker_CPU " << id << " starting doJobs()" << endl;
			}
			printStatus();
			#endif
			if (async_jobs_taken > 0) {
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " doing " << async_jobs_taken << " jobs." << endl;
				}
				printStatus();
				#endif
				doJobs();
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " finished doJobs(), creating " << last_result.load()->new_jobs.size() << " new jobs: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
				}
				printStatus();
				#endif
				//cerr << "Worker_CPU " << id << " finished beginTransaction: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
				//WriteResult(&last_result, dict, txn);
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " writing " << last_result.load()->new_jobs.size() << " new jobs." << endl;
				}
				printStatus();
				#endif
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " about to write results." << endl;
				}
				printStatus();
				#endif
				auto txn = thread_db.load()->beginTransaction();
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " began transaction." << endl;
				}
				printStatus();
				#endif
				WriteResult(last_result.load(), dict, txn.txn);
				#ifdef DEBUG_WORKER_CPU
				{
					std::lock_guard<std::mutex> lock(global_print_mutex);
					cerr << "Worker_CPU " << id << " committed " << last_result.load()->new_jobs.size() << " new jobs." << endl;
				}
				printStatus();
				#endif

				// update: workers finish their own input jobs (their input jobs only come from their own db)
				finishJobs(txn.txn);
			}

			finished = true;
		}
		cerr << "Worker_CPU " << id << " exiting loop, setting judgement_day." << endl;
		terminated = true;
		judgement_day = true;
	}
	struct loop_raii {
		Worker_CPU* worker;
		loop_raii(Worker_CPU* p_worker)
		{
			worker = p_worker;
		}
		~loop_raii()
		{
			worker->terminated = true;
			worker->judgement_day = true;
			cerr << "Worker_CPU " << worker->id << " exiting loop_raii, setting terminated and judgement_day." << endl;
			worker->printStatus();
		}
	};
	void loop()
	{
		loop_raii _raii(this);
		try {
			thread_db = new Database(main_thread_db->db_name);
			loop_internal();
		}
		catch (std::exception& e) {
			cerr << "Worker_CPU " << id << " caught exception in loop: " << e.what() << endl;
			terminated = true;
			judgement_day = true;
		}
		terminated = true;
		judgement_day = true;
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
	void terminate() override
	{
		// call base
		Worker::terminate();
		// sets terminated = true
		if (thread.joinable()) {
			cerr << "Worker " << id << " Joining thread" << endl;
			thread.join();
			cerr << "Worker " << id << " Thread joined" << endl;
		}
		else {
			cerr << "Worker " << id << " Thread not joinable" << endl;
		}
	}

	int32_t numThreads() override
	{
		#ifdef CUDANAGRAM_THREADS_PER_CPU_WORKER
		if (CUDANAGRAM_THREADS_PER_CPU_WORKER <= 0) {
			cerr << "CUDANAGRAM_THREADS_PER_CPU_WORKER is set to invalid value "
				 << CUDANAGRAM_THREADS_PER_CPU_WORKER << endl;
			throw new std::runtime_error("unspecified");
		}
		return CUDANAGRAM_THREADS_PER_CPU_WORKER;
		#endif
		return 256;  // CPU cores are ~4-5x faster than CUDA cores at sequential work
	}

	void doJob(job::Job* p_input, int64_t p_count) override
	{
		frequency_map::FrequencyMap tmp = {};
		job::Job tmp_job = {};
		tmp_job.parent_job_id = p_input->job_id;
		FrequencyMapIndex_t start = p_input->start;
		// cerr << "Worker_CPU " << id << " doJob: starting from frequency map index " << start << endl;
		// cerr << "input freuqency map: ";
		//p_input->frequency_map.print();
		FrequencyMapIndex_t end = dict.load()->frequency_maps_length;
		if (start >= end) {
			throw new std::runtime_error("unspecified");
		}

		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict.load()->h_compareFrequencyMaps_pip(
				&p_input->frequency_map,
				i,
				&tmp_job.frequency_map
			);
			#ifdef DEBUG_WORKER_CPU
			{
				cerr << "Worker_CPU " << id << " doJob: comparing frequency map: ";
				p_input->frequency_map.print();
				cerr << " with dictionary frequency map index " << i << ": ";
				dict.load()->printWordsAt(i, 1);
				cerr << " result = " << result << endl;
			}
			#endif
			if (result == INCOMPLETE_MATCH) {
				tmp_job.start = i;
				tmp_job.is_sentence = false;
				tmp_job.finished = false;
				last_result.load()->new_jobs.push_back(tmp_job);
				#ifdef DEBUG_WORKER_CPU
				{
					cerr << "Worker_CPU " << id << " doJob: spawned new job from parent "
						<< tmp_job.parent_job_id << " at frequency map index " << i << endl;
				}
				#endif
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				tmp_job.is_sentence = true;
				tmp_job.finished = true;
				last_result.load()->new_jobs.push_back(tmp_job);
				#ifdef DEBUG_WORKER_CPU
				{
					cerr << "Worker_CPU " << id << " doJob: spawned new SENTENCE job from parent "
						<< tmp_job.parent_job_id << " at frequency map index " << i << endl;
					cerr << "Worker_CPU " << id << " checking last_result.new_jobs.back(): ";
					last_result.load()->new_jobs.back().print();
				}
				#endif
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
	#ifdef CUDANAGRAM_NUM_CPU_WORKERS
		for (int64_t i = 0; i < CUDANAGRAM_NUM_CPU_WORKERS; i++) {
	#else
        for (int64_t i = 0; i < max && i < concurrent_threads; i++) {
	#endif
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
