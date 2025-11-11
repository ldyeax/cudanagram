#define NUM_JOBS_PER_BATCH 1024*512
#include "definitions.hpp"
#include "anagrammer.hpp"
#include <iostream>
#include <unistd.h>
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
using std::cout;
using std::endl;
using std::string;

using namespace anagrammer;

anagrammer::Anagrammer::Anagrammer(string p_input)
{
    input = p_input;
    dict = new Dictionary(input)
    db = new Database();
    unfinished_jobs = new Job[NUM_JOBS_PER_BATCH];
}

anagrammer::Anagrammer::Anagrammer(string p_input, Dictionary* p_dict)
{
    input = p_input;
    dict = p_dict;
    db = new Database();
    unfinished_jobs = new Job[NUM_JOBS_PER_BATCH];
}

void anagrammer::Anagrammer::run_generation()
{

}