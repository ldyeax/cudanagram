#define DEBUG_WORKER_CPU 1
#define TEST_WORKER_GPU 1
#define CUDANAGRAM_TESTING 1
#include "avx.hpp"
#include "frequency_map.cu"
#include "dictionary.cu"
#include "worker_gpu.cu"
using namespace worker_GPU;
using job::Job;
int main(int argc, char** argv)
{
	string input = "helloworld";
	Dictionary dict(
		(char*)input.c_str(),
		(char*)"./worker_cpu_test_dictionary.txt",
		nullptr,
		0
	);
	/*
	    Worker_GPU(
			int p_device_id,
			Dictionary* p_dict,
			Job* p_initial_jobs,
			int64_t p_num_initial_jobs,
			shared_ptr<vector<Job>> non_sentence_finished_jobs
		)
	*/
	Job input_job = Job();
	input_job.job_id = 1;
	input_job.parent_job_id = 0;
	dict.copyInputFrequencyMap(&input_job.frequency_map);
	input_job.start = 0;
	input_job.finished = false;
	input_job.is_sentence = false;

	shared_ptr<vector<Job>> non_sentence_finished_jobs = make_shared<vector<Job>>();

	Worker_GPU w = Worker_GPU(
		0,
		&dict,
		&input_job,
		1,
		non_sentence_finished_jobs
	);
	w.start();
	while(true);
}
