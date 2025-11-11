#pragma once
#include <cstring>
#include <stdio.h>
#include "job.hpp"
#include <string>
#include <memory>
using std::unique_ptr;
using std::vector;
using std::shared_ptr;
using std::make_shared;
namespace database {
	struct Impl;
	struct Txn;
	class Database {
	private:
		std::string getNewDatabaseName();
		std::string db_name;
		Impl* impl;
		void create_db();
		void connect();
		void init();
	public:
		Txn* beginTransaction();
		void commitTransaction(Txn* txn);
		~Database();
		Database();
		Database(std::string existing_db_name);
		// Write job to db, where its ID will be created
		void writeJob(job::Job);
		void writeJob(job::Job, Txn* txn);
		void writeJobs(job::Job* jobs, int32_t length);
		void writeJobs(job::Job* jobs, int32_t length, Txn* txn);
		/**
		 * Input should have start=index of last word in sentence
		 * Sentence will be found by traversing parent_job_id
		 * **/
		shared_ptr<vector<FrequencyMapIndex_t>> writeCompleteSentence(job::Job);
		shared_ptr<vector<FrequencyMapIndex_t>> writeCompleteSentence(job::Job, Txn* txn);
		job::Job getJob(JobID_t job_id);
		job::Job getJob(JobID_t job_id, Txn* txn);
		int32_t getUnfinishedJobs(int32_t length, job::Job*& output);
		int32_t getUnfinishedJobs(int32_t length, job::Job*& output, Txn* txn);
		void finishJobs(job::Job* jobs, int32_t length);
		void finishJobs(job::Job* jobs, int32_t length, Txn* txn);
	};
}
