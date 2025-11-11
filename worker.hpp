#pragma once

#include <memory>
#include "job.hpp"
#include "frequency_map.hpp"
#include "database.hpp"
#include "dictionary.hpp"
#include <stdint.h>

using job::Job;
namespace worker {
    struct Result {
        Job* new_jobs;
        int32_t num_new_jobs;
        Job* found_sentences;
        int32_t num_found_sentences;
    };
    class Worker {
    public:
        Result last_result;
        Worker(database::Database* db, dictionary::Dictionary* dict);
        database::Database* db;
        dictionary::Dictionary* dict;
        virtual Result doJob(Job job) = 0;
        void WriteResult(Result result);
        void WriteResult(Result result, dictionary::Dictionary* dict);
        void WriteResult();
    };
    extern Worker* workerFactory_CPU(database::Database* db, dictionary::Dictionary* dict);
    extern Worker* workerFactory_GPU(database::Database* db, dictionary::Dictionary* dict);
}
