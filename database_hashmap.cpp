#include "database.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <cstring>
#include <memory>
#include <cstdint>
#include <vector>
#include <tuple>
#include "job.hpp"
#include <arpa/inet.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <cstdio>

#include <unordered_map>
#include <mutex>

using std::cout;
using std::endl;
using job::Job;

// helper for 64-bit network byte order
static inline uint64_t htonll(uint64_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)htonl(v & 0xffffffffULL) << 32) | htonl((uint32_t)(v >> 32));
#else
    return v;
#endif
}

using namespace database;
using namespace std;
using job::Job;
using std::unique_ptr;
using std::make_unique;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

#include <atomic>
#include <thread>
static std::atomic<int64_t> max_id{100};

struct database::Impl {
	string database_name;
	std::unordered_map<JobID_t, job::Job> jobs_map;
	std::mutex map_mutex;
};
struct database::Txn {
	Txn(Impl* impl) {
		// No transaction needed for in-memory database
	}
	void commit() {
		// No commit needed for in-memory database
	}
};

std::string Database::getNewDatabaseName()
{
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return "cudanagram_" + current_unix_timestamp;
}

void sweepThread(Impl* impl)
{
	//auto file_handle = fopen("sweep_log.txt", "a");
	// remove job if it's finished and has no children
	while (true) {
		for (auto it = impl->jobs_map.begin(); it != impl->jobs_map.end(); ) {
			const Job& job = it->second;
			if (job.finished) {
				// Check if any job has this job as parent
				bool has_children = false;
				for (const auto& pair : impl->jobs_map) {
					if (pair.second.parent_job_id == job.job_id) {
						has_children = true;
						break;
					}
				}
				if (!has_children) {
					// Remove job
					it = impl->jobs_map.erase(it);
					continue;
				}
			}
			++it;
		}
	}
}

void Database::init()
{
	impl = new Impl;
	// Start sweep thread
	std::thread sweeper(sweepThread, impl);
	sweeper.detach();
}

Txn* Database::beginTransaction()
{
	return new Txn(impl);
}

void Database::commitTransaction(Txn* txn)
{
	txn->commit();
	delete txn;
}

Database::~Database()
{
	if (impl != nullptr) {
		delete impl;
		impl = nullptr;
	}
}

Database::Database(std::string existing_db_name)
{
#ifdef TEST_DB
	cerr << "Constructing Database object with existing db name: " << existing_db_name << endl;
#endif
	init();
	db_name = existing_db_name;
	connect();
#ifdef TEST_DB
	cerr << "Connected to existing db" << endl;
#endif
}
Database::Database(Database* other)
{
#ifdef TEST_DB
	cerr << "Constructing Database object with existing db name: " << other->db_name << endl;
#endif
	init();
	db_name = other->db_name;
	connect();
#ifdef TEST_DB
	cerr << "Connected to existing db" << endl;
#endif
}

void Database::create_db()
{
	db_name = getNewDatabaseName();
	impl->database_name = db_name;
	cerr << "Created new db" << endl;
}

void Database::connect()
{

}

Database::Database()
{
	init();
	create_db();
	connect();
}

void Database::writeJob(job::Job job) {
	Txn txn(impl);
	writeJobs(&job, 1, &txn);
	txn.commit();
}

void Database::writeJob(job::Job job, Txn* txn) {
	writeJobs(&job, 1, txn);
}

void Database::writeJobs(job::Job* jobs, int64_t length)
{
	Txn txn(impl);
	writeJobs(jobs, length, &txn);
	txn.commit();
}



void Database::writeJobs(job::Job* jobs, int64_t length, Txn* txn)
{
	if (length <= 0) {
		throw;
	}

	std::lock_guard<std::mutex> lock(impl->map_mutex);
	for (int64_t i = 0; i < length; i++) {
		Job& j = jobs[i];
		if (j.job_id == 0) {
			j.job_id = max_id.fetch_add(1);
		}
		impl->jobs_map[j.job_id] = j;
	}

}


void Database::finishJobs(job::Job* jobs, int64_t length) {
	Txn txn(impl);
	finishJobs(jobs, length, &txn);
	txn.commit();
}

void Database::finishJobs(job::Job* jobs, int64_t length, Txn* txn) {
    if (length <= 0)
	{
		throw;
	}

	std::lock_guard<std::mutex> lock(impl->map_mutex);
	//cerr << "Finishing " << length << " jobs" << endl;
	for (int64_t i = 0; i < length; ++i) {
		JobID_t id = jobs[i].job_id;
		auto it = impl->jobs_map.find(id);
		if (it != impl->jobs_map.end()) {
			it->second.finished = true;
			//cerr << "Finished job " << id << ": impl->jobs_map.find(id).second.finished = " << impl->jobs_map.find(id)->second.finished << endl;
		}
		else {
			throw;
		}
	}
}


void Database::printFoundSentence(
	FrequencyMapIndex_t start,
	JobID_t parent_id,
	Dictionary* dict,
	shared_ptr<vector<FrequencyMapIndex_t>> indices,
	Txn* txn
)
{
	//cerr << start << endl;

	if (parent_id != 0) {
		indices->push_back(start);
		auto it = impl->jobs_map.find(parent_id);
		if (it == impl->jobs_map.end()) {
			throw;
		}
		JobID_t next_parent_id = it->second.parent_job_id;
		FrequencyMapIndex_t next_start = it->second.start;
		printFoundSentence(
			next_start,
			next_parent_id,
			dict,
			indices,
			txn
		);
	}
	else {
		// Reached the root, print the sentence
		dict->printSentence(indices);
	}
}

void Database::printFoundSentences(Dictionary* dict)
{
	//cerr << "Printing found sentences:" << endl;
	Txn* txn = beginTransaction();
	for (auto const& pair : impl->jobs_map) {
		const Job& job = pair.second;
		if (!job.is_sentence) {
			continue;
		}
		JobID_t job_id = job.job_id;
		//cerr << "Found sentence job_id=" << job_id << endl;
		JobID_t parent_id = job.parent_job_id;
		FrequencyMapIndex_t start = job.start;
		shared_ptr<vector<FrequencyMapIndex_t>> indices = make_shared<vector<FrequencyMapIndex_t>>();
		printFoundSentence(
			start,
			parent_id,
			dict,
			indices,
			txn
		);
		// cerr << "===" << endl;
	}
	cerr << "Printed " << impl->jobs_map.size() << " found sentences" << endl;
	commitTransaction(txn);
}



int64_t Database::getUnfinishedJobs(int64_t length, Job* buffer)
{
	Txn txn(impl);
	int64_t out_count = getUnfinishedJobs(length, buffer, &txn);
	txn.commit();
	return out_count;
}

int64_t getJobCountSlow(Impl* impl) {
	return impl->jobs_map.size();
}

int64_t getUnfinishedJobCountSlow(Impl* impl) {
	int64_t count = 0;
	for (const auto& pair : impl->jobs_map) {
		if (!pair.second.finished) {
			++count;
		}
	}
	return count;
}

void Database::printJobsStats()
{
	Txn txn(impl);
	int64_t total_jobs = getJobCountSlow(impl);
	int64_t unfinished_jobs = getUnfinishedJobCountSlow(impl);
	cerr << "Jobs stats: total_jobs=" << total_jobs
		 << ", unfinished_jobs=" << unfinished_jobs << endl;
	txn.commit();
}

int64_t Database::getUnfinishedJobs(int64_t length, job::Job* buffer, Txn* txn)
{
	if (length <= 0) {
		throw;
	}

	fprintf(stderr, "Found %ld jobs, of which %ld are unfinished\n",
		getJobCountSlow(impl),
		getUnfinishedJobCountSlow(impl)
	);

	int64_t count = 0;
	for (const auto& pair : impl->jobs_map) {
		if (!pair.second.finished) {
			if (count < length) {
				buffer[count] = pair.second;
				// mark as finished immediately
				// impl->jobs_map[pair.first].finished = true;
			}
			else {
				return count;
			}
			++count;
		}
	}
	return count;
}

job::Job Database::getJob(JobID_t id)
{
	Txn txn(impl);
	job::Job ret = getJob(id, &txn);
	txn.commit();
	return ret;
}

job::Job Database::getJob(JobID_t id, Txn* txn)
{
	if (id < 1) {
		throw std::invalid_argument("invalid id in Database::getJob: " + std::to_string(id));
	}
	auto it = impl->jobs_map.find(id);
	if (it == impl->jobs_map.end()) {
		throw std::out_of_range("Job ID not found in Database::getJob: " + std::to_string(id));
	}
	return it->second;
}
