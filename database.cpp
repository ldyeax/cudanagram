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


#include <arpa/inet.h>
#include <cstdint>
#include <vector>
#include <cstring>

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
	cout << "Constructing Database object with existing db name: " << existing_db_name << endl;
#endif
	init();
	db_name = existing_db_name;
	connect();
#ifdef TEST_DB
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
#ifdef TEST_DB
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

void Database::writeUnfinishedJob(job::Job job) {
	Txn txn(impl);
	writeUnfinishedJobs(&job, 1, &txn);
	txn.commit();
}

void Database::writeUnfinishedJob(job::Job job, Txn* txn) {
	writeUnfinishedJobs(&job, 1, txn);
}

void Database::writeUnfinishedJobs(job::Job* jobs, int64_t length)
{
	Txn txn(impl);
	writeUnfinishedJobs(jobs, length, &txn);
	txn.commit();
}



void Database::writeUnfinishedJobs(job::Job* jobs, int64_t length, Txn* txn)
{
	if (length <= 0) {
		throw;
	}
	{
		pqxx::table_path job_table_path({"job"});
		auto s = pqxx::stream_to::table(*txn, job_table_path, {
			"parent_job_id",
			"frequency_map",
			"start",
			"finished"
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
				false
			);
		}

		s.complete();
	}

	{
		// std::vector<JobID_t> parent_job_id_values;
		// std::vector<FrequencyMapIndex_t> start_values;
		// std::vector<pqxx::binarystring> frequency_map_values;
		// parent_job_id_values.reserve(length);
		// start_values.reserve(length);
		// frequency_map_values.reserve(length);
		// for (int32_t i = 0; i < length; i++) {
		// 	Job& j = jobs[i];
		// 	parent_job_id_values.push_back(j.parent_job_id);
		// 	start_values.push_back(j.start);
		// 	// pqxx::binarystring fs(
		// 	// 	(const char*)j.frequency_map.asStdBytePointer(),
		// 	// 	NUM_LETTERS_IN_ALPHABET
		// 	// );
		// 	// frequency_map_values.push_back(fs);
		// 	frequency_map_values.emplace_back(
		// 		reinterpret_cast<const unsigned char*>(j.frequency_map.asStdBytePointer()),
		// 		NUM_LETTERS_IN_ALPHABET
		// 	);
		// }

		// // txn->txn->exec_params(
		// // 	"INSERT INTO job (parent_job_id, frequency_map, start, finished) "
		// // 	"SELECT UNNEST($1::BIGINT[]), UNNEST($2::BYTEA[]), UNNEST($3::INTEGER[]), FALSE",
		// // 	parent_job_id_values,
		// // 	frequency_map_values,
		// // 	start_values
		// // );
		// txn->txn->exec_params0(
		// 	"INSERT INTO job (parent_job_id, frequency_map, start, finished)\n"
		// 	"SELECT u.parent_job_id, u.frequency_map, u.start, FALSE\n"
		// 	"FROM unnest($1::bigint[], $2::bytea[], $3::int[]) AS u(parent_job_id, frequency_map, start)",
		// 	parent_job_id_values,
		// 	frequency_map_values,
		// 	start_values
		// );
	}
	// {
	// 	std::string sql;
	// 	sql.reserve(length * 128);
	// 	sql += "INSERT INTO job(parent_job_id, frequency_map, start, finished) VALUES";
	// 	for (int32_t i = 0; i < length; ++i) {
	// 		const Job& j = jobs[i];
	// 		pqxx::binarystring fm(
	// 			reinterpret_cast<const unsigned char*>(j.frequency_map.asStdBytePointer()),
	// 			NUM_LETTERS_IN_ALPHABET);
	// 		if (i) sql += ',';
	// 		sql += '('
	// 			+ pqxx::to_string(j.parent_job_id) + ','
	// 			+ txn->txn->quote(fm) + ','
	// 			+ pqxx::to_string(j.start) + ",FALSE)";
	// 	}
	// 	txn->txn->exec0(sql);
	// }
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

    std::vector<JobID_t> ids;
    ids.reserve(length);
    for (int64_t i = 0; i < length; ++i) ids.push_back(jobs[i].job_id);

    txn->txn->exec_params(
        "UPDATE job SET finished = TRUE "
        "WHERE job_id = ANY($1::BIGINT[]) AND finished IS DISTINCT FROM TRUE",
        ids
    );
}

shared_ptr<vector<FrequencyMapIndex_t>> Database::writeCompleteSentence(job::Job job)
{
	Txn txn(impl);
	auto result = writeCompleteSentence(job, &txn);
	txn.commit();
	return result;
}

shared_ptr<vector<FrequencyMapIndex_t>> Database::writeCompleteSentence(job::Job job, Txn* txn)
{
#ifdef TEST_DB
	printf("Writing complete sentence starting from job %ld\n", job.job_id);
#endif

	static bool prepared = false;
	if (!prepared) {
		impl->conn->prepare("insert_arrays", "INSERT INTO found_sentences (frequency_map_indices) VALUES ($1)");
		prepared = true;
	}

	//std::vector<FrequencyMapIndex_t> frequency_map_indices;
	shared_ptr<vector<FrequencyMapIndex_t>> frequency_map_indices
		= make_shared<vector<FrequencyMapIndex_t>>();
	frequency_map_indices->push_back(job.start);
#ifdef TEST_DB
	printf("%d ", job.start);
#endif

	while (job.parent_job_id > 0)
	{
		job = getJob(job.parent_job_id, txn);
		// Don't add start from the root job
		if (job.parent_job_id == 0) {
			break;
		}
#ifdef TEST_DB
		printf("%d ", job.start);
#endif
		frequency_map_indices->push_back(job.start);
	}
#ifdef TEST_DB
	printf("\n");
#endif

	txn->txn->exec_prepared("insert_arrays", frequency_map_indices);

	return frequency_map_indices;
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

int64_t Database::getUnfinishedJobs(int64_t length, job::Job* buffer)
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

int64_t Database::getUnfinishedJobs(int64_t length, job::Job* buffer, Txn* txn)
{
	if (length <= 0) {
		throw;
	}

	printf("Found %ld jobs, of which %ld are unfinished\n",
		getJobCountSlow(txn),
		getUnfinishedJobCountSlow(txn)
	);

	std::string query =
		std::string("SELECT "
			"job_id, "
			"parent_job_id, "
			"frequency_map, "
			"start, "
			//"parent_frequency_map_index, "
			"finished "
		"FROM job "
		"WHERE finished = false "
		"LIMIT ") + std::to_string(length);
#ifdef TEST_DB
	cout << "Executing query: " << query << endl;
#endif
	pqxx::result res = txn->txn->exec(query);
#ifdef TEST_DB
	cout << "Executed query, got " << res.size() << " results" << endl;
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
