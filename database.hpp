#pragma once
#include <cstring>
#include <stdio.h>
#include "job.hpp"
#include <string>
#include <memory>
#include "dictionary.hpp"
#include <stdio.h>
using std::unique_ptr;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using dictionary::Dictionary;
using std::cerr;
using std::endl;

namespace database {
	extern bool use_memory_db;
	extern bool gpu_memory_db;

	struct Impl;
	struct Txn;
	class DatabaseBase {
		public:
		std::string db_name;
		virtual void commitTransaction(Txn* txn) = 0;
	};

	struct TxnContainer {
		DatabaseBase* db;
		Txn* txn = nullptr;
		bool committed = false;
		TxnContainer(DatabaseBase* p_db, Txn* p_txn)
		{
			db = p_db;
			if (db == nullptr) {
				throw std::invalid_argument("TxnContainer constructed with null database");
			}
			txn = p_txn;
			if (txn == nullptr) {
				throw std::invalid_argument("TxnContainer constructed with null txn");
			}
			#ifdef SQLITE_TEST
			cerr << "TxnContainer " << db->db_name << " constructed for database " << endl;
			#endif
		}
		~TxnContainer()
		{
			if (!committed) {
				#ifdef SQLITE_TEST
				cerr << "TxnContainer " << db->db_name << " destructed without commit, committing." << endl;
				#endif
				db->commitTransaction(txn);
			}
			#ifdef SQLITE_TEST
			else {
				cerr << "TxnContainer " << db->db_name << " destructed after commit." << endl;
			}
			#endif
		}
		operator Txn*()
		{
		 	return txn;
		}
	};

	class Database : public DatabaseBase {
	private:
		std::string getNewDatabaseName();
		Impl* impl = nullptr;
		void create_db();
		void connect();
		void init();
		bool using_cache = true;
		bool memory = false;
	public:
		bool has_found_sentence;
		databaseType_t getDatabaseType();
		// char* finishJobs_buffer = nullptr;
		// int64_t finishJobs_buffer_size = 0;
		TxnContainer beginTransaction();
		virtual void commitTransaction(Txn* txn) override;
		void close();
		~Database();
		Database(bool memory);
		Database();
		Database(std::string existing_db_name);
		// Write job to db, where its ID will be created
		void writeJob(job::Job);
		void writeJob(job::Job, Txn* txn);
		void writeNewJobs(job::Job* jobs, int64_t length);
		void writeNewJobs(job::Job* jobs, int64_t length, Txn* txn);
		/**
		 * Insert jobs that have already had an ID generated
		 */
		void insertJobsWithIDs(job::Job* jobs, int64_t length);
		void insertJobsWithIDs(job::Job* jobs, int64_t length, Txn* txn);
		/**
		 * Input should have start=index of last word in sentence
		 * Sentence will be found by traversing parent_job_id
		 * **/
		// shared_ptr<vector<FrequencyMapIndex_t>> writeCompleteSentence(job::Job);
		// shared_ptr<vector<FrequencyMapIndex_t>> writeCompleteSentence(job::Job, Txn* txn);
		job::Job getJob(JobID_t job_id);
		job::Job getJob(JobID_t job_id, Txn* txn);
		int64_t getUnfinishedJobs(int64_t length, job::Job* output);
		int64_t getUnfinishedJobs(int64_t length, job::Job* output, Txn* txn);
		int64_t getUnfinishedJobs(int64_t length, vector<Job>* buffer);
		int64_t getUnfinishedJobs(int64_t length, vector<Job>* buffer, Txn* txn);
		int64_t getAndRemoveUnfinishedJobs(
			int64_t length,
			job::Job* output
		);
		// void finishJobs_startBuilding();
		// void finishJobs_add(JobID_t job_id);
		// void finishJobs_finishBuilding(Txn* txn);
		void finishJobs(job::Job* jobs, int64_t length);
		void finishJobs(job::Job* jobs, int64_t length, Txn* txn);
		void finishJobsOnSelfAndChildren(job::Job* jobs, int64_t length);
		/**
		 * Returns total unfinished jobs across self and children
		 */
		int64_t printJobsStats();
		void printFoundSentence(
			FrequencyMapIndex_t start,
			JobID_t parent_id,
			Dictionary* dict,
			shared_ptr<vector<FrequencyMapIndex_t>> indices,
			Txn* txn,
			FILE* output_file
		);
		void printFoundSentences(Dictionary* dict, FILE* output_file);
		void setJobIDIncrementStart(int64_t start);
		void getFoundSentenceJobs(vector<Job>& out_jobs);
		void getFoundSentenceJobs(vector<Job>& out_jobs, Txn* txn);
		int64_t getSentenceJobCountSlow();
	};

	extern vector<Database*> getExistingDatabases();


	// class Database_PSQL : public Database {
	// 	// PSQL-specific implementations
	// };
	// class Database_Memory : public Database {
	// 	// In-memory-specific implementations
	// };
}
