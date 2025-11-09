#include "database.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <pqxx/pqxx>
#include <pqxx/transaction>
#include <pqxx/tablewriter>
#include <pqxx/stream_to>
#include <cstring>
#include <memory>
#include <cstdint>

using namespace database;
using namespace std;
using job::Job;
using std::unique_ptr;
using std::make_unique;

struct database::Impl {
	unique_ptr<pqxx::connection> conn;
};

std::string Database::getNewDatabaseName()
{
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return "cudanagram_" + current_unix_timestamp;
}

void Database::init()
{
	char tmp[] = "PGPASSWORD=cudanagram";
	putenv(tmp);
}

Database::Database(std::string existing_db_name)
{
	init();
	db_name = existing_db_name;
	connect();
}

void Database::create_db()
{
	db_name = getNewDatabaseName();
	string tmp = "psql -U cudanagram -v dbname=";
	tmp += db_name;
	tmp += " -f setup.sql";
	if (!system(tmp.c_str())) {
		throw;
	}
}

void Database::connect()
{
	string tmp = "dbname=";
	tmp += db_name;
	tmp += " user=cudanagram host=/var/run/postgresql";
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

	static constexpr std::array<std::string_view,5> COLS{
        "parent_job_id",
		"frequency_map",
		"start",
		"parent_frequency_map_index",
		"finished"
    };
	pqxx::stream_to s{
		txn,
		"job",
		COLS
	};

	for (int32_t i = 0; i < length; i++) {
		Job& j = jobs[i];
		pqxx::binarystring fm(
			j.frequency_map,
			NUM_LETTERS_IN_ALPHABET
		);
		s.write_row(
			j.parent_job_id,
			fm,
			j.start,
			j.parent_frequency_map_index,
			j.finished
		);
	}
	s.complete();
}

void Database::writeCompleteSentence(job::Job job)
{
	vector<FrequencyMapIndex_t> frequency_map_indices = {};
	do {
		frequency_map_indices.push_back(
			job.parent_frequency_map_index
		);
	} while (job.parent_frequency_map_index >= 0);
	pqxx::work txn = pqxx::work(*impl->conn);
	auto w = pqxx::tablewriter(
		txn,
		"found_sentences", {
			"frequency_map_indices"
		}
	);
	int32_t length = frequency_map_indices.size();
	FrequencyMapIndex_t* buffer
		= new FrequencyMapIndex_t[length];
	for (int32_t i = 1; i <= length; i++) {
		buffer[length - i] = frequency_map_indices[i - 1];
	}
	w.insert({
		pqxx::binarystring(reinterpret_cast<char*>(buffer)),
		sizeof(FrequencyMapIndex_t) * length
	});
	w.complete();
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
		"SELECT "
			"job_id, "
			"job_parent_id, "
			"frequency_map, "
			"start, "
			"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE finished = false "
		"LIMIT " + length;
	pqxx::work txn = pqxx::work(*impl->conn);
	pqxx::result res = txn.exec(query);
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
		throw;
	}
	std::string query =
		"SELECT "
			"job_id, "
			"job_parent_id, "
			"frequency_map, "
			"start, "
			"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE job_id = " + id;
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
