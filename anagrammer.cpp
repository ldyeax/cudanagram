//#define NUM_JOBS_PER_BATCH 1024*512

// #define TEST_ANAGRAMMER 1

#include "definitions.hpp"
#include "anagrammer.hpp"
#include <iostream>
#include <unistd.h>
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include <stdint.h>
#include <string>
#include "definitions.hpp"
#include "dictionary.hpp"
#include "database.hpp"
#include "job.hpp"
#include "worker.hpp"
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
using dictionary::Dictionary;
using job::Job;
using std::shared_ptr;
using std::vector;
using database::Database;
using std::string;
using worker::Worker;
using worker::WorkerFactory;

using namespace anagrammer;

using std::cerr;
using std::endl;

Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input)
{
	num_jobs_per_batch = p_num_jobs_per_batch;
	input = p_input;
	dict = new Dictionary(input.data(), nullptr, nullptr, 0);
	database = new Database();
}
Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input, Dictionary* p_dict)
{
	num_jobs_per_batch = p_num_jobs_per_batch;
	input = p_input;
	dict = p_dict;
	database = new Database();
}
Anagrammer::Anagrammer(int64_t p_num_jobs_per_batch, string p_input, string p_dict_filename)
{
	num_jobs_per_batch = p_num_jobs_per_batch;
	input = p_input;
	dict = new Dictionary(input.data(), (char*)p_dict_filename.c_str(), nullptr, 0);
	database = new Database();
}
