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
        db->writeUnfinishedJobs(
            result->new_jobs.data(),
            result->new_jobs.size(),
            txn
        );
    }
	if (dict != nullptr) {
		for (int32_t i = 0; i < result->found_sentences.size(); i++) {
			dict->printSentence(
				db->writeCompleteSentence(
					result->found_sentences[i],
					txn
				)
			);
		}
	}
}
