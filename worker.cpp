#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
using namespace worker;

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

void Worker::WriteResult(Result result)
{
    if (result.num_new_jobs > 0) {
        db->writeJobs(
            result.new_jobs,
            result.num_new_jobs
        );
    }
    for (int32_t i = 0; i < result.num_found_sentences; i++) {
        db->writeCompleteSentence(
            result.found_sentences[i]
        );
    }
}

void Worker::WriteResult(Result result, dictionary::Dictionary* dict)
{
    if (result.num_new_jobs > 0) {
        db->writeJobs(
            result.new_jobs,
            result.num_new_jobs
        );
    }
    for (int32_t i = 0; i < result.num_found_sentences; i++) {
        dict->printSentence(
            db->writeCompleteSentence(
                result.found_sentences[i]
            )
        );
    }
}

void Worker::WriteResult()
{
    WriteResult(last_result);
}