/**
 * SQLite implementation of the database layer
 *
 * Table schema:
 * CREATE TABLE job (
 *     job_id INTEGER PRIMARY KEY AUTOINCREMENT,
 *     parent_job_id INTEGER,
 *     frequency_map BLOB NOT NULL CHECK(length(frequency_map) = 26),
 *     start INTEGER,
 *     finished INTEGER NOT NULL,
 *     is_sentence INTEGER NOT NULL
 * );
 */

// #define SQLITE_TEST

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
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <cstdio>
#include <sqlite3.h>
#include <stdexcept>
#include <atomic>

#include <mutex>

using std::cout;
using std::endl;
using std::cerr;
using job::Job;

using namespace database;
using namespace std;
using job::Job;
using std::unique_ptr;
using std::make_unique;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;

std::atomic<int64_t> database_id {1};

struct database::Impl {
	std::mutex mutex;
	sqlite3* db = nullptr;
	Database* parent = nullptr;
	vector<Database*> children {};
	int64_t id = 0;
	Impl() : db(nullptr)
	{
		id = database_id.fetch_add(1);
	}
	~Impl() {
		if (db) {
			sqlite3_close(db);
			db = nullptr;
		}
	}
};

struct database::Txn {
	sqlite3* db;
	bool committed;

	Impl* impl;

	Txn(Impl* p_impl) : db(p_impl->db), committed(false) {
		impl = p_impl;
		char* err_msg = nullptr;
		if (sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, &err_msg) != SQLITE_OK) {
			string error = "Failed to begin transaction: ";
			if (err_msg) {
				error += err_msg;
				sqlite3_free(err_msg);
			}
			throw std::runtime_error(error);
		}
	}

	void commit() {
		if (!committed) {
			char* err_msg = nullptr;
			if (sqlite3_exec(db, "COMMIT", nullptr, nullptr, &err_msg) != SQLITE_OK) {
				string error = "Failed to commit transaction: ";
				if (err_msg) {
					error += err_msg;
					sqlite3_free(err_msg);
				}
				throw std::runtime_error(error);
			}
			committed = true;
		}
	}

	~Txn() {
		if (!committed) {
			sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
		}
	}
};

databaseType_t Database::getDatabaseType()
{
	return DB_TYPE_SQLITE;
}

std::string Database::getNewDatabaseName()
{
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return "sqlite/cudanagram_" + current_unix_timestamp + ".db";
}

void Database::init()
{
	impl = new Impl;
}

Txn* Database::beginTransaction()
{
	return new Txn(impl);
}

void Database::commitTransaction(Txn* txn)
{
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}
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
void Database::addChild(Database* other)
{
	impl->children.push_back(other);
}
Database::Database(Database* other)
{
	init();
	static std::mutex construct_mutex;
	std::lock_guard<std::mutex> lock(construct_mutex);
#ifdef TEST_DB
	cerr << "Constructing Database object with existing db name: " << other->db_name << endl;
#endif
	other->addChild(this);
	impl->parent = other;
	create_db();
	connect();
#ifdef TEST_DB
	cerr << "Connected to existing db" << endl;
#endif
}

void Database::create_db()
{
	if (impl->parent != nullptr) {
		db_name = string(impl->parent->db_name) + string(".child.") + std::to_string(impl->id) + string(".db");
	}
	else {
		db_name = getNewDatabaseName();
	}
	cerr << "Creating SQLite db with name " << db_name << endl;

	// Open with flags to support multi-threaded access
	int rc = sqlite3_open_v2(db_name.c_str(), &impl->db,
		SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
		nullptr);
	if (rc != SQLITE_OK) {
		string error = "Cannot create database: ";
		error += sqlite3_errmsg(impl->db);
		sqlite3_close(impl->db);
		impl->db = nullptr;
		throw std::runtime_error(error);
	}

	// Enable performance optimizations and multi-threading support
	char* err_msg = nullptr;
	const char* pragmas =
		"PRAGMA page_size = 32768;"  // Larger page size for bulk operations - must be first!
		"PRAGMA journal_mode = OFF;"  // Memory journal for speed
		"PRAGMA synchronous = OFF;"
		"PRAGMA temp_store = MEMORY;"
		"PRAGMA cache_size = -16000000;"  // 16GB cache
		"PRAGMA mmap_size = 2147483648;"  // 2GB memory-mapped I/O
	;
	if (sqlite3_exec(impl->db, pragmas, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to set pragmas for database " + db_name + ": ";
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}

	// Create table
	const char* create_table_sql =
		"CREATE TABLE job ("
		"    job_id INTEGER PRIMARY KEY AUTOINCREMENT,"
		"    parent_job_id INTEGER,"
		"    frequency_map BLOB NOT NULL CHECK(length(frequency_map) = 26),"
		"    start INTEGER,"
		"    finished INTEGER NOT NULL,"
		"    is_sentence INTEGER NOT NULL"
		")";

	if (sqlite3_exec(impl->db, create_table_sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to create table: ";
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}

	// Don't create index during initial creation - it slows down bulk inserts
	// Index can be created later if needed for queries
	// const char* create_index_sql =
	// 	"CREATE INDEX idx_finished ON job(finished) WHERE finished = 0";
	// if (sqlite3_exec(impl->db, create_index_sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
	// 	string error = "Failed to create index: ";
	// 	if (err_msg) {
	// 		error += err_msg;
	// 		sqlite3_free(err_msg);
	// 	}
	// 	throw std::runtime_error(error);
	// }

	Job placeholder_job;
	placeholder_job.job_id = -impl->id;
	placeholder_job.parent_job_id = 0;
	placeholder_job.is_sentence = false;
	placeholder_job.finished = false;

	const char* insert_placeholder_job_sql =
		"INSERT INTO job (job_id, parent_job_id, frequency_map, start, finished, is_sentence) VALUES (?, ?, ?, ?, ?, ?)";
	sqlite3_stmt* stmt;
	// job_id = -impl->id, finished=true
	rc = sqlite3_prepare_v2(impl->db, insert_placeholder_job_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare insert placeholder job statement: ";
		error += sqlite3_errmsg(impl->db);
		throw std::runtime_error(error);
	}
	// sqlite3_bind_int64(stmt, 1, placeholder_job.job_id);
	// sqlite3_bind_blob(stmt, 2, placeholder_job.frequency_map.asStdBytePointer(),
	// 					NUM_LETTERS_IN_ALPHABET, SQLITE_STATIC);
	sqlite3_bind_int64(stmt, 1, -impl->id);
	sqlite3_bind_int64(stmt, 2, 0);
	sqlite3_bind_blob(stmt, 3, placeholder_job.frequency_map.asStdBytePointer(),
						NUM_LETTERS_IN_ALPHABET, SQLITE_STATIC);
	sqlite3_bind_int(stmt, 4, 0);
	sqlite3_bind_int(stmt, 5, 1);
	sqlite3_bind_int(stmt, 6, 0);
	rc = sqlite3_step(stmt);
	if (rc != SQLITE_DONE) {
		string error = "Failed to insert placeholder job:";
		error += sqlite3_errmsg(impl->db);
		sqlite3_finalize(stmt);
		throw std::runtime_error(error);
	}
	sqlite3_finalize(stmt);

	//cerr << "Created new SQLite db" << endl;
}

void Database::connect()
{
	//cerr << "Connecting to SQLite db: " << db_name << endl;

	// Open with flags to support multi-threaded access
	int rc = sqlite3_open_v2(db_name.c_str(), &impl->db,
		SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX,
		nullptr);
	if (rc != SQLITE_OK) {
		string error = "Cannot open database: ";
		error += sqlite3_errmsg(impl->db);
		sqlite3_close(impl->db);
		impl->db = nullptr;
		throw std::runtime_error(error);
	}

	// Enable performance optimizations and multi-threading support
	char* err_msg = nullptr;
	const char* pragmas =
		"PRAGMA synchronous = OFF;"
		"PRAGMA journal_mode = WAL;"  // WAL mode allows concurrent readers
		"PRAGMA temp_store = MEMORY;"
		"PRAGMA cache_size = -2000000;"  // 2GB cache
		"PRAGMA mmap_size = 2147483648;"  // 2GB memory-mapped I/O
		"PRAGMA busy_timeout = 2147483647;"; // Max int32 (~24 days) - effectively infinite

	if (sqlite3_exec(impl->db, pragmas, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to set pragmas: ";
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}
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
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}
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
	#ifdef SQLITE_TEST
	cerr << "Writing " << length << " jobs to database " << db_name;
	fprintf(stderr, " txn->db=%p\n", txn->db);
	#endif
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in writeJobs");
	}
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}

	const char* insert_sql =
		"INSERT INTO job (parent_job_id, frequency_map, start, finished, is_sentence) "
		"VALUES (?, ?, ?, ?, ?)";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, insert_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare insert statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}
#ifdef SQLITE_TEST
	cerr << "Database " << impl->id << ": Prepared insert statement for writing jobs" << endl;
#endif
	for (int64_t i = 0; i < length; i++) {
		Job& j = jobs[i];
		#ifdef SQLITE_TEST
		cerr << "Database " << impl->id << ": Writing job to database: " << endl;
		j.print();
		#endif

		// Bind parameters
		sqlite3_bind_int64(stmt, 1, j.parent_job_id);
		sqlite3_bind_blob(stmt, 2, j.frequency_map.asStdBytePointer(),
						  NUM_LETTERS_IN_ALPHABET, SQLITE_STATIC);
		sqlite3_bind_int(stmt, 3, j.start);
		sqlite3_bind_int(stmt, 4, j.finished ? 1 : 0);
		sqlite3_bind_int(stmt, 5, j.is_sentence ? 1 : 0);

		// Execute
		rc = sqlite3_step(stmt);
		if (rc != SQLITE_DONE) {
			string error = "Failed to insert job: ";
			error += sqlite3_errmsg(txn->db);
			sqlite3_finalize(stmt);
			throw std::runtime_error(error);
		}

		// Reset for next iteration
		sqlite3_reset(stmt);
	}

	sqlite3_finalize(stmt);

#ifdef SQLITE_TEST
	cerr << "Done writing " << length << " jobs to database " << db_name << endl;
#endif
}

void Database::finishJobs(job::Job* jobs, int64_t length) {
	Txn txn(impl);
	finishJobs(jobs, length, &txn);
	txn.commit();
}

void Database::finishJobs(job::Job* jobs, int64_t length, Txn* txn) {
    if (length <= 0) {
		return;
	}
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}

	const char* update_sql = "UPDATE job SET finished = 1 WHERE job_id = ?";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, update_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare update statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	for (int64_t i = 0; i < length; ++i) {
		sqlite3_bind_int64(stmt, 1, jobs[i].job_id);

		rc = sqlite3_step(stmt);
		if (rc != SQLITE_DONE) {
			string error = "Failed to update job: ";
			error += sqlite3_errmsg(txn->db);
			sqlite3_finalize(stmt);
			throw std::runtime_error(error);
		}

		sqlite3_reset(stmt);
		sqlite3_clear_bindings(stmt);
	}

	sqlite3_finalize(stmt);
}

void Database::printFoundSentence(
	FrequencyMapIndex_t start,
	JobID_t parent_id,
	Dictionary* dict,
	shared_ptr<vector<FrequencyMapIndex_t>> indices,
	Txn* txn
)
{
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}
	if (parent_id != 0) {
		indices->push_back(start);

		const char* select_sql = "SELECT parent_job_id, start FROM job WHERE job_id = ?";
		sqlite3_stmt* stmt;
		int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &stmt, nullptr);
		if (rc != SQLITE_OK) {
			string error = "Failed to prepare select statement: ";
			error += sqlite3_errmsg(txn->db);
			throw std::runtime_error(error);
		}

		sqlite3_bind_int64(stmt, 1, parent_id);

		rc = sqlite3_step(stmt);
		if (rc != SQLITE_ROW) {
			sqlite3_finalize(stmt);
			throw std::runtime_error("Expected exactly one row");
		}

		JobID_t next_parent_id = sqlite3_column_int64(stmt, 0);
		FrequencyMapIndex_t next_start = sqlite3_column_int(stmt, 1);

		sqlite3_finalize(stmt);

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
	Txn* txn = beginTransaction();

	const char* select_sql = "SELECT DISTINCT parent_job_id, start FROM job WHERE is_sentence = 1";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		error += sqlite3_errmsg(txn->db);
		delete txn;
		throw std::runtime_error(error);
	}

	int count = 0;
	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
		JobID_t parent_id = sqlite3_column_int64(stmt, 0);
		FrequencyMapIndex_t start = sqlite3_column_int(stmt, 1);

		shared_ptr<vector<FrequencyMapIndex_t>> indices = make_shared<vector<FrequencyMapIndex_t>>();
		printFoundSentence(
			start,
			parent_id,
			dict,
			indices,
			txn
		);
		count++;
	}

	sqlite3_finalize(stmt);
	cerr << "Printed " << count << " found sentences" << endl;
	commitTransaction(txn);
}

void rowToJob(sqlite3_stmt* stmt, job::Job& j)
{
	j.job_id = sqlite3_column_int64(stmt, 0);
	j.parent_job_id = sqlite3_column_int64(stmt, 1);
	j.start = sqlite3_column_int(stmt, 3);

	const void* blob = sqlite3_column_blob(stmt, 2);
	int blob_size = sqlite3_column_bytes(stmt, 2);

	std::memset(j.frequency_map.frequencies, 0, NUM_LETTERS_IN_ALPHABET);
	if (blob_size > 0) {
		std::memcpy(
			j.frequency_map.frequencies,
			blob,
			std::min(blob_size, (int)NUM_LETTERS_IN_ALPHABET)
		);
	}

	j.finished = sqlite3_column_int(stmt, 4) != 0;
}

int64_t Database::getUnfinishedJobs(int64_t length, Job* buffer)
{
	#ifdef SQLITE_TEST
	cerr << "Database::getUnfinishedJobs called with length " << length << endl;
	#endif
	Txn txn(impl);
	#ifdef SQLITE_TEST
	cerr << "Started transaction for getUnfinishedJobs" << endl;
	#endif
	int64_t out_count = getUnfinishedJobs(length, buffer, &txn);
	#ifdef SQLITE_TEST
	cerr << "Fetched " << out_count << " unfinished jobs" << endl;
	#endif
	txn.commit();
	#ifdef SQLITE_TEST
	cerr << "Committed transaction for getUnfinishedJobs" << endl;
	#endif
	return out_count;
}

int64_t getJobCountSlow(Txn* txn);

int64_t getJobCountEstimate(Txn* txn) {
	// SQLite doesn't have the same statistics as PostgreSQL
	// Just return the slow count
	return getJobCountSlow(txn);
}

int64_t getJobCountSlow(Txn* txn) {
	const char* count_sql = "SELECT COUNT(*) FROM job";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, count_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare count statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	rc = sqlite3_step(stmt);
	if (rc != SQLITE_ROW) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Failed to get count");
	}

	int64_t count = sqlite3_column_int64(stmt, 0);
	sqlite3_finalize(stmt);
	return count;
}

int64_t getUnfinishedJobCountSlow(Txn* txn) {
	const char* count_sql = "SELECT COUNT(*) FROM job WHERE finished = 0";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, count_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare count statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	rc = sqlite3_step(stmt);
	if (rc != SQLITE_ROW) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Failed to get count");
	}

	int64_t count = sqlite3_column_int64(stmt, 0);
	sqlite3_finalize(stmt);
	return count;
}

void Database::printJobsStats()
{
	Txn txn(impl);
	int64_t total_jobs = getJobCountSlow(&txn);
	int64_t unfinished_jobs = getUnfinishedJobCountSlow(&txn);
	cerr << "Database " << db_name << ": "
		 << "Jobs stats: total_jobs=" << total_jobs
		 << ", unfinished_jobs=" << unfinished_jobs << endl;
	txn.commit();
}

void Database::setJobIDIncrementStart(int64_t start)
{
	char* err_msg = nullptr;
	string seq_sql =
		"UPDATE sqlite_sequence SET seq = " + std::to_string(start) + " WHERE name = 'job';";
	#ifdef SQLITE_TEST
	cerr << "Setting job_id increment start with SQL: " << seq_sql << endl;
	#endif

	if (sqlite3_exec(impl->db, seq_sql.c_str(), nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to set job_id increment start: ";
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}

	#ifdef SQLITE_TEST
	cerr << "Set job_id increment start to " << start << endl;
	// fetch start from db
	const char* fetch_sql = "SELECT seq FROM sqlite_sequence WHERE name = 'job';";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(impl->db, fetch_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare fetch statement: ";
		error += sqlite3_errmsg(impl->db);
		throw std::runtime_error(error);
	}
	rc = sqlite3_step(stmt);
	if (rc != SQLITE_ROW) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Failed to fetch job_id increment start");
	}
	int64_t fetched_start = sqlite3_column_int64(stmt, 0);
	sqlite3_finalize(stmt);
	cerr << "Verified job_id increment start is now " << fetched_start << endl;
	#endif
}

int64_t Database::getUnfinishedJobs(int64_t length, job::Job* buffer, Txn* txn)
{
	#ifdef SQLITE_TEST
	cerr << "Database::getUnfinishedJobs called with length " << length << endl;
	#endif
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in getUnfinishedJobs");
	}
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}

	#ifdef SQLITE_TEST
	fprintf(stderr, "Database %ld: Found %ld jobs, of which %ld are unfinished\n",
		impl->id,
		getJobCountSlow(txn),
		getUnfinishedJobCountSlow(txn)
	);
	#endif

	// SQLite doesn't support UPDATE...RETURNING directly like PostgreSQL
	// We need to do it in two steps: SELECT then UPDATE

	// First, select unfinished jobs
	string select_query =
		"SELECT job_id, parent_job_id, frequency_map, start, finished "
		"FROM job "
		"WHERE finished = 0 "
		"LIMIT " + std::to_string(length);

	#ifdef SQLITE_TEST
	cerr << "Preparing select query: " << select_query << endl;
	#endif

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_query.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	std::vector<JobID_t> job_ids;
	int64_t out_count = 0;

	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW && out_count < length) {
		Job& j = buffer[out_count];
		rowToJob(stmt, j);
		#ifdef SQLITE_TEST
		cerr << "Database " << impl->id << ": Fetched unfinished job: " << endl;
		j.print();
		#endif
		job_ids.push_back(j.job_id);
		out_count++;
	}

	sqlite3_finalize(stmt);

	if (out_count < length) {
		#ifdef SQLITE_TEST
		cerr << "Fetching unfinished jobs from " << impl->children.size() << " child databases" << endl;
		cerr << "start out_count = " << out_count << endl;
		fprintf(stderr, "start buffer=%p\n", buffer);
		#endif
		for (auto child_db : impl->children) {
			#ifdef SQLITE_TEST
			cerr << "Fetching unfinished jobs from child database " << child_db->impl->id;
			fprintf(stderr, " into &buffer[out_count]=%p\n", &buffer[out_count]);
			#endif
			int64_t child_count = child_db->getUnfinishedJobs(length - out_count, &buffer[out_count]);
			#ifdef SQLITE_TEST
			cerr << "Fetched " << child_count << " unfinished jobs from child database "
				 << child_db->impl->id << endl;
			#endif
			out_count += child_count;
			if (out_count >= length) {
				break;
			}
		}
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
		throw std::invalid_argument("invalid id: " + std::to_string(id));
	}
	if (txn->db != impl->db) {
		throw std::invalid_argument("Transaction does not belong to this database");
	}

	const char* select_sql =
		"SELECT job_id, parent_job_id, frequency_map, start, finished "
		"FROM job "
		"WHERE job_id = ?";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	sqlite3_bind_int64(stmt, 1, id);

	rc = sqlite3_step(stmt);
	if (rc != SQLITE_ROW) {
		sqlite3_finalize(stmt);
		for (auto child_db : impl->children) {
			try {
				return child_db->getJob(id);
			} catch (const std::runtime_error&) {
				// Ignore and try next child
			}
		}
		throw std::runtime_error("Job not found with id: " + std::to_string(id));
	}

	job::Job ret;
	rowToJob(stmt, ret);

	sqlite3_finalize(stmt);
	return ret;
}
