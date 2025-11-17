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
	Worker_CPU(database::Database* db, dictionary::Dictionary* dict, int32_t p_id)
	{
		id.store(p_id);
	}
};

class WorkerFactory_CPU : public WorkerFactory {
public:
	int64_t Spawn(
		Worker** buffer,
		Dictionary* dict,
		int64_t initial_job_id,
		int64_t job_ids_per_worker
	) override
	{
		unsigned int num_threads
			= std::thread::hardware_concurrency();
		if (concurrent_threads < 2) {
			throw;
		}
		concurrent_threads--;
		std::cerr << "Spawning " << concurrent_threads << " CPU workers" << endl;
		for (unsigned int i = 0; i < num_threads; i++) {
			std::thread t1([&]{
				buffer[i] = new Worker_CPU(
					initial_job_id + job_ids_per_worker * i,

	}
};

WorkerFactory* worker::getWorkerFactory_CPU(database::Database* db, dictionary::Dictionary* dict)
{
	static WorkerFactory* ret = new WorkerFactory_CPU();
	return ret;
}
