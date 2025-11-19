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

// #define TEST_DB

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

bool database::use_memory_db = false;
bool database::gpu_memory_db = false;

struct database::Impl {
	//std::mutex mutex;
	sqlite3* db = nullptr;
	Database* parent = nullptr;
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
	//std::unique_lock<std::mutex> lock;

	Txn(Impl* p_impl) :
			db(p_impl->db),
			committed(false)
			//lock(p_impl->mutex, std::defer_lock)

	{
		//bool got_lock = false;
		// {
		// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
		// 	cerr << "Txn: Acquiring DB mutex lock for database id " << p_impl->id << endl;
		// }
		try {
			//got_lock = lock.try_lock();
			// if (got_lock) {
			// 	{
			// 		//std::lock_guard<std::mutex> lock(global_print_mutex);
			// 		cerr << "Txn: Got lock for " << p_impl->id << endl;
			// 	}
			// }
		} catch (...) {
			{
				//std::lock_guard<std::mutex> lock(global_print_mutex);
				cerr << "Txn: Failed to acquire DB mutex lock for database id " << p_impl->id << endl;
			}
			throw new std::runtime_error("Failed to acquire DB mutex lock");
		}
		// if (!got_lock) {
		// 	{
		// 		//std::lock_guard<std::mutex> lock(global_print_mutex);
		// 		cerr << "Txn: Failed to acquire DB mutex lock for database id " << p_impl->id << endl;
		// 	}
		// 	throw new std::runtime_error("Failed to acquire DB mutex lock");
		// }
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
		// {
		// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
		// 	cerr << "Began transaction on database id " << impl->id << endl;
		// }
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
			// //sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
			// {
			// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
			// 	cerr << "~Txn: Transaction on database id " << impl->id << " not committed, committing." << endl;
			// }
			commit();
		}
		// else {
		// 	{
		// 		//std::lock_guard<std::mutex> lock(global_print_mutex);
		// 		cerr << "~Txn: Transaction on database id " << impl->id << " already committed." << endl;
		// 	}
		// }
		// Ensure lock released
		//lock.unlock();
		// {
		// 	//std::lock_guard<std::mutex> lock(global_print_mutex);
		// 	cerr << "~Txn: Released DB mutex lock for database id " << impl->id << endl;
		// }
	}
};

void Database::checkTxn(Txn* txn) {
	if (txn->db != impl->db) {
		fprintf(
			stderr,
			"Transaction database pointer %p does not match database %p\n",
			(void*)txn->db,
			(void*)impl->db
		);
		throw std::invalid_argument("Transaction does not belong to this database");
	}
}

databaseType_t Database::getDatabaseType()
{
	return DB_TYPE_SQLITE;
}

std::string Database::getNewDatabaseName()
{
	if (memory) {
		return ":memory:";
	}
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return "sqlite/cudanagram_" + current_unix_timestamp + ".db";
}

void Database::init()
{
	impl = new Impl;
	fprintf(stderr, "Initialized database impl at %p for database at %p\n", (void*)impl, (void*)this);
}

TxnContainer Database::beginTransaction()
{
	TxnContainer ret(this, new Txn(impl));
	return ret;
}

void Database::commitTransaction(Txn* txn)
{
	//cerr << "commitTransaction called on database " << db_name << endl;
	checkTxn(txn);
	txn->commit();  // Just commit, don't delete - let TxnContainer handle deletion
}

void Database::close()
{
	if (impl != nullptr) {
		if (impl->db != nullptr) {
			sqlite3_close(impl->db);
			impl->db = nullptr;
		}
	}
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
throw;
#ifdef TEST_DB
	cerr << "Constructing Database object with existing db name: " << existing_db_name << endl;
#endif
	db_name = existing_db_name;
	init();
	connect();
	cerr << "Connected to existing db" << endl;
#ifdef TEST_DB
#endif
}

const char* sqlite_db_pragmas =
	"PRAGMA page_size = 32768;"  // Larger page size for bulk operations - must be first!
	"PRAGMA journal_mode = OFF;"  // No journal for maximum speed
	"PRAGMA synchronous = OFF;"  // No fsync - data loss possible on crash
	"PRAGMA temp_store = MEMORY;"  // Keep temp tables in memory
	"PRAGMA cache_size = -2000000;"  // 2GB cache
	"PRAGMA mmap_size = 2147483648;"  // 2GB memory-mapped I/O
	"PRAGMA locking_mode = EXCLUSIVE;"  // No lock contention
	"PRAGMA auto_vacuum = NONE;"  // Disable auto-vacuum overhead
	"PRAGMA count_changes = OFF;"  // Don't count changes
	"PRAGMA query_only = OFF;"  // Allow writes
	"PRAGMA read_uncommitted = ON;"  // Allow dirty reads (single connection so safe)
;
void Database::create_db()
{
	if (impl->db != nullptr) {
		fprintf(
			stderr,
			"Database %s already has an open connection, cannot create new database\n",
			db_name.c_str()
		);
		throw std::runtime_error("Database already created or connected");
	}
	if (impl->parent != nullptr) {
		throw;
		db_name = string(impl->parent->db_name) + string(".child.") + std::to_string(impl->id) + string(".db");
	}
	else {
		db_name = getNewDatabaseName();
	}
	//cerr << "Creating SQLite db with name " << db_name << endl;

	// Open with flags to support multi-threaded access
	int rc = sqlite3_open_v2(db_name.c_str(), &impl->db,
		SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
		nullptr);
	if (rc != SQLITE_OK) {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		string error = "Cannot create database: ";
		cerr << error;
		const char* to_add = sqlite3_errmsg(impl->db);
		if (to_add == NULL) {
			cerr << " (no error message available)" << endl;
		}
		else {
			cerr << to_add << " ";
			error += to_add;
		}
		sqlite3_close(impl->db);
		impl->db = nullptr;
		throw std::runtime_error(error);
	}
	else {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Created SQLite db: " << db_name << endl;
	}

	// Enable performance optimizations and multi-threading support
	char* err_msg = nullptr;
	// allow multithreaded

	if (sqlite3_exec(impl->db, sqlite_db_pragmas, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		string error = "Failed to set pragmas for database " + db_name + ": ";
		cerr << error;
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}
	else {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Set pragmas for database " << db_name << endl;
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
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << error;
		if (err_msg) {
			cerr << err_msg;
			error += err_msg;
			sqlite3_free(err_msg);
		}
		else {
			cerr << " (no error message available)" << endl;
		}
		throw std::runtime_error(error);
	}
	else {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		#if TEST_DB
		cerr << "Created job table in database " << db_name << endl;
		#endif
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

	//cerr << "Created new SQLite db" << endl;
}

void Database::connect()
{
	#ifdef TEST_DB
	{
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Connecting to SQLite db: " << db_name << endl;
	}
	#endif

	// Open with flags to support multi-threaded access
	int rc = sqlite3_open_v2(db_name.c_str(), &impl->db,
		SQLITE_OPEN_READWRITE ,
		nullptr);
	if (rc != SQLITE_OK) {
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		string error = "Cannot open database: ";
		cerr << error;
		const char* to_add = sqlite3_errmsg(impl->db);
		if (to_add == NULL) {
			cerr << " (no error message available)" << endl;
		}
		else {
			cerr << to_add << " ";
		}
		sqlite3_close(impl->db);
		impl->db = nullptr;
		{
			//std::lock_guard<std::mutex> lock(global_print_mutex);
			cerr << error << endl;
		}
		throw std::runtime_error(error);
	}

	{
		//std::lock_guard<std::mutex> lock(global_print_mutex);
		cerr << "Connected to SQLite db: " << db_name << endl;
	}

	// Enable performance optimizations and multi-threading support
	char* err_msg = nullptr;

	if (sqlite3_exec(impl->db, sqlite_db_pragmas, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to set pragmas: ";
		cerr << error << endl;
		if (err_msg) {
			cerr << err_msg << endl;
			error += err_msg;
			sqlite3_free(err_msg);
		}
		else {
			cerr << " (no error message available)" << endl;
		}
		throw std::runtime_error(error);
	}
}

Database::Database(bool p_memory)
{
	memory = p_memory;
	init();
	create_db();
}
Database::Database()
{
	fprintf(stderr, "Creating database at %p\n", (void*)this);
	memory = use_memory_db;
	init();
	create_db();
	fprintf(stderr, "Database() impl->db = %p\n", (void*)impl->db);
	// Don't call connect() for in-memory databases - connection is already open from create_db()
	// connect() would create a NEW empty in-memory database
}

void Database::writeJob(job::Job job) {
	Txn txn(impl);
	writeNewJobs(&job, 1, &txn);
	txn.commit();
}

void Database::writeJob(job::Job job, Txn* txn) {
checkTxn(txn);
	writeNewJobs(&job, 1, txn);
}

void Database::writeNewJobs(job::Job* jobs, int64_t length)
{
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in writeNewJobs");
	}
	Txn txn(impl);
	writeNewJobs(jobs, length, &txn);
	txn.commit();
}

/**
 * Write "new jobs", which may either be unfinished or finished (could be sentences)
 */
void Database::writeNewJobs(job::Job* jobs, int64_t length, Txn* txn)
{
	#ifdef TEST_DB
	cerr << "Writing " << length << " jobs to database " << db_name;
	fprintf(stderr, " txn->db=%p\n", txn->db);
	#endif
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in writeNewJobs");
	}
checkTxn(txn);

	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);

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
#ifdef TEST_DB
	bool any_sentence = false;
#endif
#ifdef TEST_DB
	cerr << "Database " << impl->id << ": Prepared insert statement for writing jobs length=" << length << endl;
#endif
	for (int64_t i = 0; i < length; i++) {
		Job& j = jobs[i];
		#ifdef TEST_DB
		cerr << "Writing job.." << endl;
		cerr << "Database " << impl->id << ": Writing job to database: " << endl;
		fprintf(stderr, "Job at %p\n", (void*)&j);
		fflush(stderr);
		j.print();
		#endif

		#ifdef TEST_DB
		if (j.frequency_map.isAllZero() && !j.finished) {
			cerr << "writeNewJobs: all-zero frequency map in job to be written to database " << db_name << endl;
			j.print();
			throw new std::runtime_error("all-zero frequency map in job");
		}

		if (j.job_id == j.parent_job_id) {
			cerr << "writeNewJobs: job " << j.job_id << " has same job_id and parent_job_id" << endl;
			j.print();
			throw new std::runtime_error("unspecified");
		}
		#endif

		// Bind parameters
		sqlite3_bind_int64(stmt, 1, j.parent_job_id);
		sqlite3_bind_blob(stmt, 2, j.frequency_map.asStdBytePointer(),
						  NUM_LETTERS_IN_ALPHABET, SQLITE_STATIC);
		sqlite3_bind_int(stmt, 3, j.start);
		sqlite3_bind_int(stmt, 4, j.finished ? 1 : 0);
		sqlite3_bind_int(stmt, 5, j.is_sentence ? 1 : 0);
		#ifdef TEST_DB
		if (j.is_sentence) {
			for (int32_t iii = 0; iii < 10; iii++)
				cerr << "Writing sentence job: " << endl;
			j.print();
			any_sentence = true;
			has_found_sentence = true;
		}
		#endif

		// Execute
		rc = sqlite3_step(stmt);
		if (rc != SQLITE_DONE) {
			string error = "Failed to insert job: ";
			error += sqlite3_errmsg(txn->db);
			sqlite3_finalize(stmt);
			throw std::runtime_error(error);
		}
		else {
			#if TEST_DB
			cerr << "Database " << impl->id << ": Inserted job with new job_id " << sqlite3_last_insert_rowid(txn->db) << endl;
			#endif
		}

		// Reset for next iteration
		sqlite3_reset(stmt);
	}
	#if TEST_DB
	cerr << "Database " << impl->id << ": Finished writing jobs, finalizing statement" << endl;
	#endif

	sqlite3_finalize(stmt);
#ifdef TEST_DB
cerr << "Database " << impl->id << ": Finished writing " << length << " jobs to database" << endl;
	if (!any_sentence) {
		cerr << "  (no sentences)" << endl;
	}
	else {
		// Fetch all sentence jobs
		const char* select_sql = "SELECT job_id, parent_job_id, frequency_map, start, finished, is_sentence FROM job WHERE is_sentence = 1 ORDER BY job_id DESC";
		sqlite3_stmt* select_stmt;
		int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &select_stmt, nullptr);
		if (rc != SQLITE_OK) {
			string error = "Failed to prepare select statement: ";
			error += sqlite3_errmsg(txn->db);
			throw std::runtime_error(error);
		}
		cerr << "  Sentence jobs in database " << impl->id << ":" << endl;
		while ((rc = sqlite3_step(select_stmt)) == SQLITE_ROW) {
			Job sjob;
			sjob.job_id = sqlite3_column_int64(select_stmt, 0);
			sjob.parent_job_id = sqlite3_column_int64(select_stmt, 1);
			const void* freq_map_blob = sqlite3_column_blob(select_stmt, 2);
			memcpy((void*)sjob.frequency_map.asStdBytePointer(), freq_map_blob, NUM_LETTERS_IN_ALPHABET);
			sjob.start = sqlite3_column_int(select_stmt, 3);
			sjob.finished = sqlite3_column_int(select_stmt, 4) != 0;
			sjob.is_sentence = sqlite3_column_int(select_stmt, 5) != 0;
			cerr << "    Job ID: " << sjob.job_id << ", Parent ID: " << sjob.parent_job_id << ", Start: " << sjob.start << ", Finished: " << sjob.finished << ", Is Sentence: " << sjob.is_sentence << endl;
		}
		sqlite3_finalize(select_stmt);
		cerr << "Finished printing sentence jobs from database " << impl->id << endl;
	}
#endif

#ifdef TEST_DB
	cerr << "Done writing " << length << " jobs to database " << db_name << endl;
#endif
}

void Database::insertJobsWithIDs(job::Job* jobs, int64_t length) {
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in insertJobsWithIDs(jobs, length)");
	}
	Txn txn(impl);
	insertJobsWithIDs(jobs, length, &txn);
	txn.commit();
}

void Database::insertJobsWithIDs(job::Job* jobs, int64_t length, Txn* txn) {
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in insertJobsWithIDs(jobs, length, txn)");
	}
checkTxn(txn);

	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);

	const char* insert_sql =
		"INSERT INTO job (job_id, parent_job_id, frequency_map, start, finished, is_sentence) "
		"VALUES (?, ?, ?, ?, ?, ?)";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, insert_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare insert statement: ";
		cerr << error;
		const char* errmsg = sqlite3_errmsg(txn->db);
		if (errmsg != NULL) {
			cerr << errmsg << endl;
			error += errmsg;
		}
		else {
			cerr << " (no error message available)" << endl;
		}
		throw std::runtime_error(error);
	}

	for (int64_t i = 0; i < length; i++) {
		Job& j = jobs[i];

		#if CUDANAGRAM_TESTING
		for (int64_t jj = 0; jj < i; jj++) {
			if (jobs[jj].job_id == j.job_id) {
				cerr << "insertJobsWithIDs: duplicate job ID " << j.job_id << " at index " << i << endl;
				throw new std::runtime_error("unspecified");
			}
			if (!jobs[jj].finished) {
				if (jobs[jj].frequency_map.isAllZero()) {
					cerr << "insertJobsWithIDs: job " << jobs[jj].job_id << " has all-zero frequency map" << endl;
					throw new std::runtime_error("unspecified");
				}
				if (jobs[jj].frequency_map.anyNegative()) {
					cerr << "insertJobsWithIDs: job " << jobs[jj].job_id << " has negative frequency map value" << endl;
					throw new std::runtime_error("unspecified");
				}
			}
		}

		if (!j.finished) {
			if (j.frequency_map.isAllZero()) {
				cerr << "insertJobsWithIDs: job " << j.job_id << " has all-zero frequency map" << endl;
				throw new std::runtime_error("unspecified");
			}
			if (j.frequency_map.anyNegative()) {
				cerr << "insertJobsWithIDs: job " << j.job_id << " has negative frequency map value" << endl;
				throw new std::runtime_error("unspecified");
			}
		}

		if (j.job_id == j.parent_job_id) {
			cerr << "insertJobsWithIDs: job " << j.job_id << " has same job_id and parent_job_id" << endl;
			throw new std::runtime_error("unspecified");
		}

		#endif

		// Bind parameters
		sqlite3_bind_int64(stmt, 1, j.job_id);
		sqlite3_bind_int64(stmt, 2, j.parent_job_id);
		sqlite3_bind_blob(stmt, 3, j.frequency_map.asStdBytePointer(),
						  NUM_LETTERS_IN_ALPHABET, SQLITE_STATIC);
		sqlite3_bind_int(stmt, 4, j.start);
		sqlite3_bind_int(stmt, 5, j.finished ? 1 : 0);
		sqlite3_bind_int(stmt, 6, j.is_sentence ? 1 : 0);

		// Execute
		rc = sqlite3_step(stmt);
		if (rc != SQLITE_DONE) {
			string error = "Failed to insert job with ID: ";
			error += sqlite3_errmsg(txn->db);
			sqlite3_finalize(stmt);
			throw std::runtime_error(error);
		}

		// Reset for next iteration
		sqlite3_reset(stmt);
	}

	sqlite3_finalize(stmt);

	#if CUDANAGRAM_TESTING
	for (int64_t i = 0; i < length; i++) {
		Job& j = jobs[i];
		if (!j.finished) {
			if (j.frequency_map.isAllZero()) {
				cerr << "insertJobsWithIDs: job " << j.job_id << " has all-zero frequency map after insert" << endl;
				throw new std::runtime_error("unspecified");
			}
			if (j.frequency_map.anyNegative()) {
				cerr << "insertJobsWithIDs: job " << j.job_id << " has negative frequency map value after insert" << endl;
				throw new std::runtime_error("unspecified");
			}
			Job from_db;
			from_db = getJob(j.job_id, txn);
			if (!from_db.frequency_map.equals(j.frequency_map)) {
				cerr << "insertJobsWithIDs: job " << j.job_id << " frequency map mismatch after insert" << endl;
				throw new std::runtime_error("unspecified");
			}
		}
	}
	cerr << "Testing passed" << endl;
	#endif
}

void Database::finishJobs(job::Job* jobs, int64_t length) {
	Txn txn(impl);
	finishJobs(jobs, length, &txn);
	txn.commit();
}

void Database::finishJobsOnSelfAndChildren(job::Job* jobs, int64_t length) {
	if (length <= 0) {
		return;
	}
	TxnContainer txn = beginTransaction();

	finishJobs(jobs, length, txn.txn);
}

void Database::finishJobs(job::Job* jobs, int64_t length, Txn* txn) {
    if (length <= 0) {
		return;
	}
checkTxn(txn);

	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);

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
	Txn* txn,
	FILE* output_file
)
{
	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
checkTxn(txn);
	if (impl->parent != nullptr) {
		throw std::invalid_argument("printFoundSentence should be called on the parent database");
	}
	if (parent_id != 0) {
		indices->push_back(start);

		// cerr << "Did push back start " << start << " for parent_id " << parent_id << endl;

		Job found_job = getJob(parent_id, txn);
		if (found_job.job_id == found_job.parent_job_id) {
			cerr << "oedipus: found job with ID " << parent_id << " in printFoundSentence" << endl;
			indices->push_back(found_job.start);
			dict->printSentence(output_file, indices);
			return;
		}

		printFoundSentence(
			found_job.start,
			found_job.parent_job_id,
			dict,
			indices,
			txn,
			output_file
		);
	}
	else {
		// Reached the root, print the sentence
		dict->printSentence(output_file, indices);
	}
}

void Database::getFoundSentenceJobs(vector<Job>& out_jobs, Txn* txn)
{
	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	//const char* select_sql = "SELECT parent_job_id, start FROM job WHERE is_sentence = 1";

	const char* select_sql = "UPDATE job SET is_sentence = -1 WHERE is_sentence = 1 RETURNING parent_job_id, start";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
		JobID_t parent_id = sqlite3_column_int64(stmt, 0);
		FrequencyMapIndex_t start = sqlite3_column_int(stmt, 1);
		// Just add to the vector
		Job j;
		j.job_id = -1; // unknown
		j.parent_job_id = parent_id;
		j.start = start;
		#ifdef TEST_DB
		cerr << "Found sentence job: parent_id=" << parent_id << ", start=" << start << endl;
		#endif
		out_jobs.push_back(j);
	}
	#ifdef TEST_DB
	cerr << "Exiting getFoundSentenceJobs with " << out_jobs.size() << " jobs found" << endl;
	#endif

	sqlite3_finalize(stmt);
}
void Database::getFoundSentenceJobs(vector<Job>& out_jobs)
{
	//cerr << "database " << db_name << " getting found sentence jobs" << endl;
	TxnContainer txn = beginTransaction();
	//cerr << "database " << db_name << " began transaction for getting found sentence jobs" << endl;
	getFoundSentenceJobs(out_jobs, txn.txn);
}

void Database::printFoundSentences(Dictionary* dict, FILE* output_file)
{
	if (impl->parent != nullptr) {
		throw std::invalid_argument("printFoundSentences should be called on the parent database");
	}
	TxnContainer txn = beginTransaction();

	vector<Job> found_sentence_jobs;
	getFoundSentenceJobs(found_sentence_jobs, txn.txn);
	//cerr << "Found " << found_sentence_jobs.size() << " sentence jobs in parent database " << db_name << endl;

	for (const auto& job : found_sentence_jobs) {
		//cerr << "Printing found sentence for job with parent_id=" << job.parent_job_id << ", start=" << job.start << endl;
		shared_ptr<vector<FrequencyMapIndex_t>> indices = make_shared<vector<FrequencyMapIndex_t>>();
		//fprintf(stderr, "Calling printFoundSentence with start=%d, parent_id=%lld, indices=%p\n", job.start, (long long)job.parent_job_id, indices.get());
		printFoundSentence(
			job.start,
			job.parent_job_id,
			dict,
			indices,
			txn.txn,
			output_file
		);
	}

	//cerr << "Committing transaction after printing found sentences" << endl;
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
	#ifdef TEST_DB
	cerr << "Database::getUnfinishedJobs called with length " << length << endl;
	#endif
	Txn txn(impl);
	#ifdef TEST_DB
	cerr << "Started transaction for getUnfinishedJobs" << endl;
	#endif
	int64_t out_count = getUnfinishedJobs(length, buffer, &txn);
	#ifdef TEST_DB
	cerr << "Fetched " << out_count << " unfinished jobs" << endl;
	#endif
	txn.commit();
	#ifdef TEST_DB
	cerr << "Committed transaction for getUnfinishedJobs" << endl;
	#endif
	return out_count;
}

int64_t getJobCountSlow(Impl* impl);
int64_t getUnfinishedJobCountSlow(Impl* impl);

int64_t getJobCountSlow(Impl* impl) {
	////lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	const char* count_sql = "SELECT COUNT(*) FROM job WHERE job_id >= 0 AND start >= 0";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(impl->db, count_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare count statement: ";
		error += sqlite3_errmsg(impl->db);
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

int64_t getUnfinishedJobCountSlow(Impl* impl) {
	////lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	const char* count_sql = "SELECT COUNT(*) FROM job WHERE job_id >= 0 AND start >= 0 AND finished = 0";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(impl->db, count_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare count statement: ";
		error += sqlite3_errmsg(impl->db);
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

int64_t _getSentenceJobCountSlow(Impl* impl);

int64_t Database::getSentenceJobCountSlow()
{
	return _getSentenceJobCountSlow(impl);
}

int64_t _getSentenceJobCountSlow(Impl* impl) {
	////lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	const char* count_sql = "SELECT COUNT(*) FROM job WHERE job_id >= 0 AND start >= 0 AND is_sentence = 1";
	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(impl->db, count_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare count statement: ";
		error += sqlite3_errmsg(impl->db);
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

int64_t Database::printJobsStats()
{
	Txn txn(impl);
	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	cerr << "Printing job stats for database " << db_name << endl;
	int64_t total_jobs = getJobCountSlow(impl);
	int64_t unfinished_jobs = getUnfinishedJobCountSlow(impl);
	int64_t sentence_jobs = _getSentenceJobCountSlow(impl);

	cerr << "Database " << db_name << ": "
		 << "Jobs stats: total_jobs=" << total_jobs
		 << ", unfinished_jobs=" << unfinished_jobs
		 << ", sentence_jobs=" << sentence_jobs << endl;
	txn.commit();
	return unfinished_jobs;
}

void removeAllJobsWithStartNegative(Impl* impl) {
	////lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	char* err_msg = nullptr;
	const char* delete_sql = "DELETE FROM job WHERE start < 0";
	if (sqlite3_exec(impl->db, delete_sql, nullptr, nullptr, &err_msg) != SQLITE_OK) {
		string error = "Failed to delete jobs with start < 0: ";
		if (err_msg) {
			error += err_msg;
			sqlite3_free(err_msg);
		}
		throw std::runtime_error(error);
	}
}

void Database::setJobIDIncrementStart(int64_t start)
{
	Job sqliteisfuckingstupid;
	sqliteisfuckingstupid.start = -1;
	writeJob(sqliteisfuckingstupid);

	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	char* err_msg = nullptr;
	string seq_sql =
		"UPDATE sqlite_sequence SET seq = " + std::to_string(start) + " WHERE name = 'job';";
	#ifdef TEST_DB
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

	#ifdef TEST_DB
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

	removeAllJobsWithStartNegative(impl);
}

int64_t Database::getUnfinishedJobs(int64_t length, job::Job* buffer, Txn* txn)
{
	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	#ifdef TEST_DB
	cerr << "Database::getUnfinishedJobs called with length " << length << endl;
	#endif
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in getUnfinishedJobs");
	}
checkTxn(txn);

	string select_query =
		"UPDATE job "
		"SET finished = 1 "
		"WHERE finished = 0 "
		"RETURNING job_id, parent_job_id, frequency_map, start, finished "
		"LIMIT " + std::to_string(length);


	#ifdef TEST_DB
	cerr << "Preparing select query: " << select_query << endl;
	#endif

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_query.c_str(), -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		error += sqlite3_errmsg(txn->db);
		throw std::runtime_error(error);
	}

	//std::vector<JobID_t> job_ids;
	int64_t out_count = 0;

	while ((rc = sqlite3_step(stmt)) == SQLITE_ROW && out_count < length) {
		Job& j = buffer[out_count];
		rowToJob(stmt, j);
		#ifdef TEST_DB
		cerr << "Database " << impl->id << ": Fetched unfinished job: " << endl;
		j.print();
		#endif
		//job_ids.push_back(j.job_id);
		out_count++;
	}

	sqlite3_finalize(stmt);

	// if (out_count < length) {
	// 	#ifdef TEST_DB
	// 	cerr << "Fetching unfinished jobs from " << impl->children.size() << " child databases" << endl;
	// 	cerr << "start out_count = " << out_count << endl;
	// 	fprintf(stderr, "start buffer=%p\n", buffer);
	// 	#endif
	// 	for (auto child_db : impl->children) {
	// 		#ifdef TEST_DB
	// 		cerr << "Fetching unfinished jobs from child database " << child_db->impl->id;
	// 		fprintf(stderr, " into &buffer[out_count]=%p\n", &buffer[out_count]);
	// 		#endif
	// 		int64_t child_count = child_db->getUnfinishedJobs(length - out_count, &buffer[out_count]);
	// 		#ifdef TEST_DB
	// 		cerr << "Fetched " << child_count << " unfinished jobs from child database "
	// 			 << child_db->impl->id << endl;
	// 		#endif
	// 		out_count += child_count;
	// 		if (out_count >= length) {
	// 			break;
	// 		}
	// 	}
	// }

	return out_count;
}

int64_t Database::getUnfinishedJobs(int64_t length, vector<Job>* buffer)
{
	#ifdef TEST_DB
	cerr << "Database::getUnfinishedJobs called with length " << length << endl;
	#endif
	Txn txn(impl);
	#ifdef TEST_DB
	cerr << "Started transaction for getUnfinishedJobs" << endl;
	#endif
	int64_t out_count = getUnfinishedJobs(length, buffer, &txn);
	#ifdef TEST_DB
	cerr << "Fetched " << out_count << " unfinished jobs" << endl;
	#endif
	txn.commit();
	#ifdef TEST_DB
	cerr << "Committed transaction for getUnfinishedJobs" << endl;
	#endif
	return out_count;
}

int64_t Database::getUnfinishedJobs(int64_t length, vector<Job>* buffer, Txn* txn)
{
	//lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	#ifdef TEST_DB
	cerr << "Database::getUnfinishedJobs called with length " << length << " and buffer with size " << buffer->size() << endl;
	#endif
	if (length <= 0) {
		throw std::invalid_argument("Invalid length in getUnfinishedJobs");
	}
checkTxn(txn);

	// SQLite doesn't support UPDATE...RETURNING directly like PostgreSQL
	// We need to do it in two steps: SELECT then UPDATE

	// First, select unfinished jobs
	// string select_query =
	// 	"SELECT job_id, parent_job_id, frequency_map, start, finished "
	// 	"FROM job "
	// 	"WHERE finished = 0 "
	// 	//"ORDER BY job_id ASC "
	// 	"LIMIT " + std::to_string(length);

	// sqlite now supports UPDATE...RETURNING
	string select_query =
		"UPDATE job "
		"SET finished = 1 "
		"WHERE finished = 0 AND job_id > 0 AND start >= 0 "
		"RETURNING job_id, parent_job_id, frequency_map, start, finished "
		"LIMIT " + std::to_string(length);


	#ifdef TEST_DB
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
		buffer->push_back(Job());
		Job& j = buffer->back();
		rowToJob(stmt, j);
		#ifdef TEST_DB
		cerr << "Database " << impl->id << ": Fetched unfinished job: " << endl;
		j.print();
		#endif
		job_ids.push_back(j.job_id);
		out_count++;
	}

	sqlite3_finalize(stmt);

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
	////lockguardtest_lock_guard<std::mutex> lock(impl->mutex);
	if (id < 1) {
		throw std::invalid_argument("invalid id: " + std::to_string(id));
	}
checkTxn(txn);

	const char* select_sql =
		"SELECT job_id, parent_job_id, frequency_map, start, finished "
		"FROM job "
		"WHERE job_id = ?";

	sqlite3_stmt* stmt;
	int rc = sqlite3_prepare_v2(txn->db, select_sql, -1, &stmt, nullptr);
	if (rc != SQLITE_OK) {
		string error = "Failed to prepare select statement: ";
		cerr << error;
		const char* errmsg = sqlite3_errmsg(txn->db);
		if (errmsg) {
			cerr << errmsg << endl;
			error += errmsg;
		} else {
			error += "unknown error";
			cerr << "Unknown error in sqlite3_prepare_v2" << endl;
		}
		throw std::runtime_error(error);
	}

	sqlite3_bind_int64(stmt, 1, id);

	rc = sqlite3_step(stmt);
	if (rc != SQLITE_ROW) {
		sqlite3_finalize(stmt);
		throw std::runtime_error("Job not found with id: " + std::to_string(id));
	}

	job::Job ret;
	rowToJob(stmt, ret);

	sqlite3_finalize(stmt);
	return ret;
}
#include <filesystem>
vector<Database*> database::getExistingDatabases()
{
	// search filesystem for .db files under ./sqlite/
	vector<Database*> dbs;
	string db_dir = "sqlite/";
	cerr << "Searching for database files in directory: " << db_dir << endl;
	for (const auto& entry : std::filesystem::directory_iterator(db_dir)) {
		cerr << "Checking file: " << entry.path().string() << endl;
		if (entry.is_regular_file()) {
			cerr << "Got regular file: " << entry.path().string() << endl;
			string path = entry.path().string();
			if (path.size() >= 3 && path.substr(path.size() - 3) == ".db") {
				cerr << "Got .db file: " << path << endl;
				// found a .db file
				try {
					cerr << "Constructing database:" << path << endl;
					Database* db = new Database(path);
					dbs.push_back(db);
				} catch (const std::exception& e) {
					cerr << "Failed to open database file " << path << ": " << e.what() << endl;
				}
			}
		}
	}
	return dbs;
}
