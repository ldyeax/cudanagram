#define TEST_DB 1

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

using namespace database;
using namespace std;
using job::Job;
using std::unique_ptr;
using std::make_unique;
using std::cout;
using std::endl;

struct database::Impl {
	unique_ptr<pqxx::connection> conn;
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

Database::~Database()
{
	if (impl != nullptr) {
		delete impl;
		impl = nullptr;
	}
}

Database::Database(std::string existing_db_name)
{
#if TEST_DB
	cout << "Constructing Database object with existing db name: " << existing_db_name << endl;
#endif
	init();
	db_name = existing_db_name;
	connect();
#if TEST_DB
	cout << "Connected to existing db" << endl;
#endif
}

void Database::create_db()
{
	db_name = getNewDatabaseName();
	cout << "Creating db with name " << db_name << endl;
	string tmp = "psql -d postgres -U cudanagram -v dbname=";
	tmp += db_name;
	tmp += " -f setup.sql";
	cout << "Executing command: " << tmp << endl;
	if (system(tmp.c_str())) {
		throw;
	}
	cout << "Created new db" << endl;
}

void Database::connect()
{
	string tmp = "dbname=";
	tmp += db_name;
	tmp += " user=cudanagram host=/var/run/postgresql";
#if TEST_DB
	cout << "Connecting to db: " << tmp << endl;
	printf("Impl=%p\n", impl);
#endif
	impl->conn = make_unique<pqxx::connection>(tmp.c_str());
}

Database::Database()
{
	init();
	create_db();
	connect();
}

void Database::writeJobs(job::Job* jobs, int32_t length)
{
	if (length <= 0) {
		throw;
	}

	pqxx::work txn = pqxx::work(*impl->conn);

	pqxx::table_path job_table_path({"job"});
	auto s = pqxx::stream_to::table(txn, job_table_path, {
		"parent_job_id",
		"frequency_map",
		"start",
		"parent_frequency_map_index",
		"finished"
	});

	for (int32_t i = 0; i < length; i++) {
		Job& j = jobs[i];

		// std::string fm(
		// 	reinterpret_cast<char*>(j.frequency_map.frequencies),
		// 	NUM_LETTERS_IN_ALPHABET
		// );
		// pqxx::binarystring fm(
		// 	reinterpret_cast<char*>(j.frequency_map.frequencies),
		// 	NUM_LETTERS_IN_ALPHABET
		// );
		pqxx::bytes_view fm(
			j.frequency_map.asStdBytePointer(),
			NUM_LETTERS_IN_ALPHABET
		);
		s.write_values(
			j.parent_job_id,
			fm,
			j.start,
			j.parent_frequency_map_index,
			j.finished
		);
	}

	s.complete();
	txn.commit();
}

void Database::writeCompleteSentence(job::Job job)
{
	// vector<FrequencyMapIndex_t> frequency_map_indices = {};
	// do {
	// 	frequency_map_indices.push_back(
	// 		job.parent_frequency_map_index
	// 	);
	// } while (job.parent_frequency_map_index >= 0);
	// pqxx::work txn = pqxx::work(*impl->conn);
	
	// std::vector<std::string> cols{"frequency_map_indices"};
	// pqxx::stream_to s{txn, "found_sentences", cols};
	
	// int32_t length = frequency_map_indices.size();
	// FrequencyMapIndex_t* buffer
	// 	= new FrequencyMapIndex_t[length];
	// for (int32_t i = 1; i <= length; i++) {
	// 	buffer[length - i] = frequency_map_indices[i - 1];
	// }
	// pqxx::binarystring bs(
	// 	reinterpret_cast<char*>(buffer),
	// 	sizeof(FrequencyMapIndex_t) * length
	// );
	// s << std::make_tuple(bs);
	// s.complete();
	// txn.commit();
	// delete[] buffer;
}

void rowToJob(const pqxx::row* p_row, job::Job& j)
{
	auto row = *p_row;
	j.job_id = row["job_id"].as<JobID_t>();
	j.parent_job_id = row["parent_job_id"].as<JobID_t>();
	j.start = row["start"].as<FrequencyMapIndex_t>();
	j.parent_frequency_map_index = row["parent_frequency_map_index"].as<FrequencyMapIndex_t>();
	pqxx::binarystring freq(row["frequency_map"]);
	std::memset(j.frequency_map.frequencies, 0, NUM_LETTERS_IN_ALPHABET);
	std::memcpy(
		j.frequency_map.frequencies,
		freq.data(),
		NUM_LETTERS_IN_ALPHABET
	);
	j.finished = row["finished"].as<bool>();
}

job::Job* Database::getUnfinishedJobs(int32_t length)
{
	if (length <= 0) {
		throw;
	}
	std::string query =
		std::string("SELECT "
			"job_id, "
			"parent_job_id, "
			"frequency_map, "
			"start, "
			"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE finished = false "
		"LIMIT ") + std::to_string(length);
#if TEST_DB
	cout << "Executing query: " << query << endl;
#endif
	pqxx::work txn = pqxx::work(*impl->conn);
#if TEST_DB
	cout << "Created transaction" << endl;
#endif
	pqxx::result res = txn.exec(query);
#if TEST_DB
	cout << "Executed query, got " << res.size() << " results" << endl;
#endif
	int32_t out_count = res.size();
	if (out_count == 0) {
		return nullptr;
	}
	job::Job* buffer = new Job[out_count];
	std::size_t i = 0;
	for (auto const &row : res) {
		Job& j = buffer[i++];
		rowToJob(&row, j);
	}
	return buffer;
}

job::Job Database::getJob(JobID_t id)
{
	if (id < 1) {
		throw std::invalid_argument("invalid id");
	}
	std::string query =
		std::string("SELECT "
			"job_id, "
			"parent_job_id, "
			"frequency_map, "
			"start, "
			"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE job_id = ") + std::to_string(id);
	pqxx::work txn = pqxx::work(*impl->conn);
	pqxx::result res = txn.exec(query);
	int32_t out_count = res.size();
	if (out_count != 1) {
		throw;
	}
	job::Job ret;
	auto row = res[0];
	rowToJob(&row, ret);
	return ret;
}
