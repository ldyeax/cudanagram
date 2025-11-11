#pragma once
#define NUM_JOBS_PER_BATCH 1024*1024
#include "definitions.hpp"
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include <stdint.h>
#include <string>
using dictionary::Dictionary;
using job::Job;
using std::shared_ptr;
using std::vector;
using database::Database;
using std::string;
namespace anagrammer {
    class Anagrammer {
    public:
        Dictionary* dict;
        Database* db;
        string input;
        Job* unfinished_jobs;
        Anagrammer(string p_input);
        Anagrammer(string p_input, Dictionary* p_dict);
        void run_generation();
    }
}