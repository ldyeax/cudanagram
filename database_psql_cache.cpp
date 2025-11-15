#include "database.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <pqxx/pqxx>
#include <pqxx/transaction>
#include <pqxx/stream_to>
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
#include <mutex>
#include <unordered_map>

namespace memory_database {
	static std::atomic<int64_t> max_id{100};
	std::unordered_map<JobID_t, job::Job> jobs_map;
	std::mutex map_mutex;
	class MemoryDatabase {
		Database* db;
		void writeNewJobsjob::Job* jobs, int64_t length)
		{
			writeNewJobsjobs, length);
		}
		void writeNewJobsjob::Job* jobs, int64_t length)
		{
			if (length <= 0) {
				throw;
			}

			std::lock_guard<std::mutex> lock(map_mutex);
			for (int64_t i = 0; i < length; i++) {
				Job& j = jobs[i];
				if (j.job_id == 0) {
					j.job_id = max_id.fetch_add(1);
				}
				jobs_map[j.job_id] = j;
			}
		}
		int64_t getJobCountSlow(Impl* impl) {
			return jobs_map.size();
		}

		int64_t getUnfinishedJobCountSlow(Impl* impl) {
			int64_t count = 0;
			for (const auto& pair : jobs_map) {
				if (!pair.second.finished) {
					++count;
				}
			}
			return count;
		}
		int64_t getUnfinishedJobs(int64_t length, job::Job* buffer, Txn* txn)
		{
			if (length <= 0) {
				throw;
			}
			fprintf(stderr, "Found %ld jobs, of which %ld are unfinished\n",
				getJobCountSlow(),
				getUnfinishedJobCountSlow()
			);

			int64_t count = 0;
			for (const auto& pair : jobs_map) {
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
	};
}


struct database::Impl {
	unique_ptr<pqxx::connection> conn;
};
struct database::Txn {
	pqxx::work* txn;
	operator pqxx::work*() {
		return txn;
	}
	operator pqxx::transaction_base&() {
		return *txn;
	}
	Txn(Impl* impl) {
		txn = new pqxx::work(*impl->conn);
	}
	void commit() {
		txn->commit();
	}
};

std::string Database::getNewDatabaseName()
{
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return "cudanagram_" + current_unix_timestamp;
}
char pgpassword_putenv[] = "PGPASSWORD=cudanagram";
void Database::init()
{
	impl = new Impl;
	putenv(pgpassword_putenv);
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
	cerr << "Creating db with name " << db_name << endl;
	string tmp = "psql -d postgres -U cudanagram -v dbname=";
	tmp += db_name;
	tmp += " -f setup.sql";
	cerr << "Executing command: " << tmp << endl;
	if (system(tmp.c_str())) {
		throw;
	}
	cerr << "Created new db" << endl;
}

void Database::connect()
{
	string tmp = "dbname=";
	tmp += db_name;
	tmp += " user=cudanagram host=/var/run/postgresql";
#ifdef TEST_DB
	cerr << "Connecting to db: " << tmp << endl;
	fprintf(stderr, "Impl=%p\n", impl);
#endif
	impl->conn = make_unique<pqxx::connection>(tmp.c_str());
}

Database::Database()
{
	init();
	create_db();
	connect();
}

void Database::writeJob(job::Job job) {
	Txn txn(impl);
	writeNewJobs(&job, 1, &txn);
	txn.commit();
}

void Database::writeJob(job::Job job, Txn* txn) {
	writeNewJobs(&job, 1, txn);
}

void Database::writeNewJobsjob::Job* jobs, int64_t length)
{
	Txn txn(impl);
	writeNewJobs(jobs, length, &txn);
	txn.commit();
}



void Database::writeNewJobs(job::Job* jobs, int64_t length, Txn* txn)
{
	if (length <= 0) {
		throw;
	}

	pqxx::table_path job_table_path({"job"});
	auto s = pqxx::stream_to::table(*txn, job_table_path, {
		"parent_job_id",
		"frequency_map",
		"start",
		"finished",
		"is_sentence"
	});

	for (int32_t i = 0; i < length; i++) {
		Job& j = jobs[i];

		pqxx::bytes_view fm(
			j.frequency_map.asStdBytePointer(),
			NUM_LETTERS_IN_ALPHABET
		);
		s.write_values(
			j.parent_job_id,
			fm,
			j.start,
			j.finished,
			j.is_sentence
		);
	}

	s.complete();

}

// void Database::finishJobs_startBuilding() {
// 	if (finishJobs_buffer != nullptr) {
// 		delete[] finishJobs_buffer;
// 		finishJobs_buffer = nullptr;
// 		finishJobs_buffer_size = 0;
// 	}
// 	finishJobs_buffer = new char[1024 * 1024]; // 1 MB initial buffer
// 	finishJobs_buffer_size = 0;
// }

void Database::finishJobs(job::Job* jobs, int64_t length) {
	// stub for psql
	// Txn txn(impl);
	// finishJobs(jobs, length, &txn);
	// txn.commit();
}

void Database::finishJobs(job::Job* jobs, int64_t length, Txn* txn) {
	// stub for psql
    // if (length <= 0)
	// {
	// 	throw;
	// }

    // std::vector<JobID_t> ids;
    // ids.reserve(length);
    // for (int64_t i = 0; i < length; ++i) ids.push_back(jobs[i].job_id);

    // txn->txn->exec_params(
    //     "UPDATE job SET finished = TRUE "
    //     "WHERE job_id = ANY($1::BIGINT[])",
    //     ids
    // );
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
		pqxx::result res = txn->txn->exec(
			"SELECT parent_job_id, start FROM job WHERE job_id = " + std::to_string(parent_id)
		);
		if (res.size() != 1) {
			throw;
		}
		JobID_t next_parent_id = res[0]["parent_job_id"].as<JobID_t>();
		FrequencyMapIndex_t next_start = res[0]["start"].as<FrequencyMapIndex_t>();
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
	pqxx::result res = txn->txn->exec(
		"SELECT job_id, parent_job_id, start FROM job WHERE is_sentence = TRUE"
	);
	for (auto const &row : res) {
		JobID_t job_id = row["job_id"].as<JobID_t>();
		//cerr << "Found sentence job_id=" << job_id << endl;
		JobID_t parent_id = row["parent_job_id"].as<JobID_t>();
		FrequencyMapIndex_t start = row["start"].as<FrequencyMapIndex_t>();
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
	cerr << "Printed " << res.size() << " found sentences" << endl;
	commitTransaction(txn);
}

void rowToJob(const pqxx::row* p_row, job::Job& j)
{
	auto row = *p_row;
	j.job_id = row["job_id"].as<JobID_t>();
	j.parent_job_id = row["parent_job_id"].as<JobID_t>();
	j.start = row["start"].as<FrequencyMapIndex_t>();
	//j.parent_frequency_map_index = row["parent_frequency_map_index"].as<FrequencyMapIndex_t>();
	//pqxx::binarystring freq(row["frequency_map"]);
	auto freq = row["frequency_map"].as<pqxx::bytes>();
	std::memset(j.frequency_map.frequencies, 0, NUM_LETTERS_IN_ALPHABET);
	std::memcpy(
		j.frequency_map.frequencies,
		freq.data(),
		NUM_LETTERS_IN_ALPHABET
	);
	j.finished = row["finished"].as<bool>();
}

int64_t Database::getUnfinishedJobs(int64_t length, Job* buffer)
{
	Txn txn(impl);
	int64_t out_count = getUnfinishedJobs(length, buffer, &txn);
	txn.commit();
	return out_count;
}

int64_t getJobCountEstimate(Txn* txn) {
	// SELECT reltuples::bigint AS estimate FROM pg_class where relname = 'mytable';

	pqxx::result res = txn->txn->exec(
		"SELECT reltuples::bigint AS estimate "
		"FROM pg_class "
		"WHERE relname = 'job'"
	);
	return res[0]["estimate"].as<int64_t>();
}

int64_t getJobCountSlow(Txn* txn) {
	pqxx::result res = txn->txn->exec(
		"SELECT COUNT(*) AS count "
		"FROM job"
	);
	return res[0]["count"].as<int64_t>();
}

int64_t getUnfinishedJobCountSlow(Txn* txn) {
	pqxx::result res = txn->txn->exec(
		"SELECT COUNT(*) AS count "
		"FROM job "
		"WHERE finished = false"
	);
	return res[0]["count"].as<int64_t>();
}

void Database::printJobsStats()
{
	Txn txn(impl);
	int64_t total_jobs = getJobCountSlow(&txn);
	int64_t unfinished_jobs = getUnfinishedJobCountSlow(&txn);
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
		getJobCountSlow(txn),
		getUnfinishedJobCountSlow(txn)
	);

	std::string query =
	std::string(
		"WITH selected AS ( "
		"    SELECT job_id, parent_job_id, frequency_map, start, finished "
		"    FROM job "
		"    WHERE finished = false "
		"    LIMIT " + std::to_string(length) + " "
		") "
		"UPDATE job "
		"SET finished = true "
		"FROM selected "
		"WHERE job.job_id = selected.job_id "
		"RETURNING selected.job_id, selected.parent_job_id, selected.frequency_map, selected.start, selected.finished"
	);

#ifdef TEST_DB
	cerr << "Executing query: " << query << endl;
#endif
	pqxx::result res = txn->txn->exec(query);
#ifdef TEST_DB
	cerr << "Executed query, got " << res.size() << " results" << endl;
#endif
	int64_t out_count = res.size();
	if (out_count == 0) {
		return 0;
	}
	std::size_t i = 0;
	for (auto const &row : res) {
		Job& j = buffer[i++];
		rowToJob(&row, j);
	}
	return out_count;
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
	std::string query =
		std::string("SELECT "
			"job_id, "
			"parent_job_id, "
			"frequency_map, "
			"start, "
			//"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE job_id = ") + std::to_string(id);

	pqxx::result res = txn->txn->exec(query);
	int32_t out_count = res.size();
	if (out_count != 1) {
		throw;
	}
	job::Job ret;
	auto row = res[0];
	rowToJob(&row, ret);
	return ret;
}
