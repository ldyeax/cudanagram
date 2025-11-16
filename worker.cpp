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
using std::cerr;
using std::endl;


Worker::Worker(database::Database* p_db, dictionary::Dictionary* p_dict)
{
    if (p_db == nullptr) {
        throw new std::runtime_error("unspecified");
    }
    if (p_dict == nullptr) {
        throw new std::runtime_error("unspecified");
    }
    main_thread_db = new Database(p_db);
    dict = p_dict;
}

void Worker::WriteResult(Result* result, dictionary::Dictionary* dict, Txn* txn)
{
	if (result == nullptr) {
		result = last_result.load();
	}
	if (txn == nullptr) {
		throw "no txn";
	}

	// fprintf(stderr,
	// 	"%d WriteResult result=%p dict=%p txn=%p db=%p \n",
	// 	id,         (void*)result, (void*)dict, (void*)txn, (void*)thread_db);
	// fprintf(stderr, "%d result->new_jobs.size() = %ld\n", id, result->new_jobs.size());
	// fprintf(stderr, "%d result->new_jobs.data() = %p\n", id, (void*)result->new_jobs.data());
	if (result->new_jobs.size() > 0) {
		thread_db.load()->writeNewJobs(
			result->new_jobs.data(),
			result->new_jobs.size(),
			txn
		);
	}
}

void Worker::doJobs()
{
	last_result.load()->new_jobs.reserve(unfinished_jobs.load()->size() * 100); // Reserve space for efficiency
	last_result.load()->new_jobs.clear();
	for (int32_t i = 0; i < (int32_t)unfinished_jobs.load()->size(); i++) {
		#ifdef DEBUG_WORKER_CPU
		cerr << "Worker " << id << " doJobs: doing job " << i << " / " << unfinished_jobs.load()->size() << endl;
		unfinished_jobs.load()->at(i).print();
		#endif
		doJob(&unfinished_jobs.load()->at(i), 1);
	}
}

void Worker::doJobs_async() {
	finished = false;
	ready_to_start = true;
}

void Worker::finishJobs() {
	thread_db.load()->finishJobs(
		last_result.load()->new_jobs.data(),
		last_result.load()->new_jobs.size()
	);
}

void Worker::finishJobs(Txn* txn) {
	thread_db.load()->finishJobs(
		last_result.load()->new_jobs.data(),
		last_result.load()->new_jobs.size(),
		txn
	);
}

void Worker::finishJobs_async() {
	// Implementation for asynchronous finishJobs can be added here
}


int64_t Worker::takeJobs(Job* buffer, int64_t max_length)
{
	if (max_length <= 0) {
		throw new std::runtime_error("unspecified");
	}
	int64_t jobs_to_take = std::min((int64_t)numThreads(), max_length);
	for (int64_t i = 0; i < jobs_to_take; i++) {
		unfinished_jobs.load()->push_back(buffer[i]);
	}
	return jobs_to_take;
}
int64_t Worker::takeJobsAndWrite(Job* buffer, int64_t max_length)
{
	if (max_length <= 0) {
		cerr << "Worker " << id << " takeJobsAndWrite: max_length <= 0" << endl;
		throw new std::runtime_error("unspecified");
	}
	int64_t jobs_to_take = std::min((int64_t)numThreads(), max_length);
	if (jobs_to_take <= 0) {
		cerr << "Worker " << id << " takeJobsAndWrite: jobs_to_take <= 0" << endl;
		throw new std::runtime_error("unspecified");
	}
	for (int64_t i = 0; i < jobs_to_take; i++) {
		if (buffer[i].frequency_map.isAllZero()) {
			cerr << "Worker " << id << " takeJobsAndWrite: all-zero frequency map in job " << buffer[i].job_id << endl;
			throw new std::runtime_error("unspecified");
		}
		if (buffer[i].frequency_map.anyNegative()) {
			cerr << "Worker " << id << " takeJobsAndWrite: negative frequency map value in job " << buffer[i].job_id << endl;
			throw new std::runtime_error("unspecified");
		}
		unfinished_jobs.load()->push_back(buffer[i]);
	}
	thread_db.load()->insertJobsWithIDs(
		buffer,
		jobs_to_take
	);
	return jobs_to_take;
}

int64_t Worker::takeJobs(int64_t max_length)
{
	if (max_length <= 0) {
		throw new std::runtime_error("unspecified");
	}

	int64_t jobs_to_take = std::min((int64_t)numThreads(), max_length);

#ifdef CUDANAGRAM_TESTING
	cerr << "Worker_CPU " << id << " takeJobs: trying to take up to "
			<< jobs_to_take << " jobs from its database." << endl;
	for (int64_t i = 0; i < unfinished_jobs.load()->size(); i++) {
		if (unfinished_jobs.load()->at(i).frequency_map.isAllZero()) {
			cerr << "Worker_CPU " << id << " takeJobs.0: all-zero frequency map in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("unspecified");
		}
		if (unfinished_jobs.load()->at(i).frequency_map.anyNegative()) {
			cerr << "Worker_CPU " << id << " takeJobs.0: negative frequency map value in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("unspecified");
		}
	}
	cerr << "Worker_CPU " << id << " passed test 1." << endl;
#endif
	unfinished_jobs.load()->reserve(unfinished_jobs.load()->size() + jobs_to_take);
	int64_t initial_size = unfinished_jobs.load()->size();
#ifdef CUDANAGRAM_TESTING
	for (int64_t i = 0; i < initial_size; i++) {
		if (unfinished_jobs.load()->at(i).frequency_map.isAllZero()) {
			cerr << "Worker_CPU " << id << " takeJobs.2: all-zero frequency map in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("unspecified");
		}
		if (unfinished_jobs.load()->at(i).frequency_map.anyNegative()) {
			cerr << "Worker_CPU " << id << " takeJobs.2: negative frequency map value in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("unspecified");
		}
	}
	cerr << "Worker_CPU " << id << " passed test 2." << endl;
	cerr << "initial_size = " << initial_size << endl;
#endif
	int64_t found = thread_db.load()->getUnfinishedJobs(
		jobs_to_take,
		unfinished_jobs.load()
	);

#ifdef CUDANAGRAM_TESTING
	for (int64_t i = 0; i < initial_size + found; i++) {
		if (unfinished_jobs.load()->at(i).frequency_map.isAllZero()) {
			unfinished_jobs.load()->at(i).print();
			cerr << "Worker_CPU " << id << " takeJobs.3: all-zero frequency map in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("all-zero frequency map in job");
		}

		if (unfinished_jobs.load()->at(i).frequency_map.anyNegative()) {
			unfinished_jobs.load()->at(i).print();
			cerr << "Worker_CPU " << id << " takeJobs.3: negative frequency map value in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("negative frequency map value in job");
		}
	}
	cerr << "Worker_CPU " << id << " passed test 3." << endl;
#endif


	#if CUDANAGRAM_TESTING
	for (int64_t i = 0; i < unfinished_jobs.load()->size(); i++) {
		if (unfinished_jobs.load()->at(i).frequency_map.isAllZero()) {
			unfinished_jobs.load()->at(i).print();
			cerr << "Worker_CPU " << id << " takeJobs: all-zero frequency map in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("all-zero frequency map in job");
		}
		if (unfinished_jobs.load()->at(i).frequency_map.anyNegative()) {
			unfinished_jobs.load()->at(i).print();
			cerr << "Worker_CPU " << id << " takeJobs: negative frequency map value in job " << unfinished_jobs.load()->at(i).job_id << endl;
			throw new std::runtime_error("negative frequency map value in job");
		}
	}
	cerr << "Worker_CPU " << id << " took " << found << " jobs from its database." << endl;
	#endif
	return found;
}

void Worker::reset()
{
	finished = false;
	ready_to_start = false;
	ready_to_take_jobs = false;
	async_jobs_to_take = 0;
	async_jobs_taken = 0;

	unfinished_jobs.load()->clear();
	last_result.load()->new_jobs.clear();
}

void Worker::terminate()
{
	terminated = true;
}
