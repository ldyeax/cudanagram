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

void Database::writeJob(job::Job job) {
	Txn txn(impl);
	writeJobs(&job, 1, &txn);
	txn.commit();
}

void Database::writeJob(job::Job job, Txn* txn) {
	writeJobs(&job, 1, txn);
}

void Database::writeJobs(job::Job* jobs, int32_t length)
{
	Txn txn(impl);
	writeJobs(jobs, length, &txn);
	txn.commit();
}

void Database::writeJobs(job::Job* jobs, int32_t length, Txn* txn)
{
	if (length <= 0) {
		throw;
	}

	pqxx::table_path job_table_path({"job"});
	auto s = pqxx::stream_to::table(*txn, job_table_path, {
		"parent_job_id",
		"frequency_map",
		"start",
		//"parent_frequency_map_index",
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
			//j.parent_frequency_map_index,
			j.finished
		);
	}

	s.complete();
}

void Database::finishJobs(job::Job* jobs, int32_t length) {
	Txn txn(impl);
	finishJobs(jobs, length, &txn);
	txn.commit();
}

void Database::finishJobs(job::Job* jobs, int32_t length, Txn* txn) {
    if (length <= 0)
	{
		throw;
	}

    std::vector<JobID_t> ids;
    ids.reserve(length);
    for (int32_t i = 0; i < length; ++i) ids.push_back(jobs[i].job_id);

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

int32_t Database::getUnfinishedJobs(int32_t length, job::Job*& buffer)
{
	Txn txn(impl);
	int32_t out_count = getUnfinishedJobs(length, buffer, &txn);
	txn.commit();
	return out_count;
}

int32_t Database::getUnfinishedJobs(int32_t length, job::Job*& buffer, Txn* txn)
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
	int32_t out_count = res.size();
	if (out_count == 0) {
		return 0;
	}
	buffer = new Job[out_count];
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
