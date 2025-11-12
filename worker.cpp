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
	Database thread_db = Database(db);
	while (true) {
		while (!ready_to_start) {
			//std::this_thread::yield();
			std::this_thread::sleep_for(std::chrono::microseconds(250000));
		}
		ready_to_start = false;
		finished = false;
		//cout << "Worker_CPU " << id << " starting doJobs(): ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		doJobs();
		//cout << "Worker_CPU " << id << " finished doJobs(), creating " << last_result.new_jobs.size() << " new jobs: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		auto txn = thread_db.beginTransaction();
		//cout << "Worker_CPU " << id << " finished beginTransaction: ready_to_start=" << ready_to_start << " finished= " << finished << endl;
		WriteResult(&last_result, dict, txn);
		//cout << "Worker_CPU " << id << " committed " << last_result.new_jobs.size() << " new jobs." << endl;
		// thread_db.finishJobs(
		// 	last_result.new_jobs.data(),
		// 	last_result.new_jobs.size(),
		// 	txn
		// );
		thread_db.commitTransaction(txn);
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
    db = p_db;
    dict = p_dict;
}

void Worker::WriteResult(Result* result, dictionary::Dictionary* dict, Txn* txn)
{
	if (result == nullptr) {
		result = &last_result;
	}
	if (db == nullptr) {
		throw "no db";
	}
	if (txn == nullptr) {
		throw "no txn";
	}
	// printf("WriteResult result=%p dict=%p txn=%p db=%p txn=%p\n", (void*)result, (void*)dict, (void*)txn, (void*)db);
	// printf("result->new_jobs.size() = %ld\n", result->new_jobs.size());
	// printf("result->new_jobs.data() = %p\n", (void*)result->new_jobs.data());
    if (result->new_jobs.size() > 0) {
        db->writeJobs(
            result->new_jobs.data(),
            result->new_jobs.size(),
            txn
        );
    }
	// if (dict != nullptr) {
	// 	for (int32_t i = 0; i < result->found_sentences.size(); i++) {
	// 		dict->printSentence(
	// 			db->writeCompleteSentence(
	// 				result->found_sentences[i],
	// 				txn
	// 			)
	// 		);
	// 	}
	// }
}

void Worker::doJobs()
{
	last_result.new_jobs = vector<Job>{unfinished_jobs.size()};
	//last_result.found_sentences = vector<Job>{unfinished_jobs.size()};
	for (int32_t i = 0; i < unfinished_jobs.size(); i++) {
		doJob(*unfinished_jobs[i]);
	}
}

void Worker::doJobs_async() {
	finished = false;
	ready_to_start = true;
}
