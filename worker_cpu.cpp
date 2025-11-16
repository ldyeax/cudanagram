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
