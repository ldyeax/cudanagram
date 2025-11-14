#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include <thread>

using namespace worker;

using job::Job;
using database::Database;
using database::Txn;
using std::vector;

void Worker::loop()
{
	while (true) {
		while (!ready_to_start) {
			//std::this_thread::yield();
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
		ready_to_start = false;
		finished = false;
		//cerr << "Worker_CPU " << id << " starting doJobs(): ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		doJobs();
		//cerr << "Worker_CPU " << id << " finished doJobs(), creating " << last_result.new_jobs.size() << " new jobs: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		auto txn = thread_db->beginTransaction();
		//cerr << "Worker_CPU " << id << " finished beginTransaction: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		WriteResult(&last_result, dict, txn);
		//cerr << "Worker_CPU " << id << " committed " << last_result.new_jobs.size() << " new jobs." << endl;
		// thread_db.finishJobs(
		// 	last_result.new_jobs.data(),
		// 	last_result.new_jobs.size(),
		// 	txn
		// );
		thread_db->commitTransaction(txn);
		finished = true;
	}
}

Worker::Worker(database::Database* p_db, dictionary::Dictionary* p_dict)
{
    if (p_db == nullptr) {
        throw;
    }
    if (p_dict == nullptr) {
        throw;
    }
    thread_db = new Database(p_db);
    dict = p_dict;
}

void Worker::WriteResult(Result* result, dictionary::Dictionary* dict, Txn* txn)
{
	if (result == nullptr) {
		result = &last_result;
	}
	if (txn == nullptr) {
		throw "no txn";
	}
	// fprintf(stderr, "WriteResult result=%p dict=%p txn=%p db=%p txn=%p\n", (void*)result, (void*)dict, (void*)txn, (void*)db);
	// fprintf(stderr, "result->new_jobs.size() = %ld\n", result->new_jobs.size());
	// fprintf(stderr, "result->new_jobs.data() = %p\n", (void*)result->new_jobs.data());
	if (result->new_jobs.size() > 0) {
		thread_db->writeJobs(
			result->new_jobs.data(),
			result->new_jobs.size(),
			txn
		);
	}
}

void Worker::doJobs()
{
	last_result.new_jobs.clear();
	last_result.new_jobs.reserve(unfinished_jobs.size() * 100); // Reserve space for efficiency
	for (int32_t i = 0; i < unfinished_jobs.size(); i++) {
		doJob(unfinished_jobs[i], 1);
	}
}

void Worker::doJobs_async() {
	finished = false;
	ready_to_start = true;
}
