#pragma once
#include <cstring>
#include <stdio.h>
#include "job.hpp"
#include <string>
#include <memory>
#include "dictionary.hpp"
using std::unique_ptr;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using dictionary::Dictionary;
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
		// char* finishJobs_buffer = nullptr;
		// int64_t finishJobs_buffer_size = 0;
		Txn* beginTransaction();
		void commitTransaction(Txn* txn);
		~Database();
		Database();
		Database(std::string existing_db_name);
		Database(Database* other);
		// Write job to db, where its ID will be created
		void writeUnfinishedJob(job::Job);
		void writeUnfinishedJob(job::Job, Txn* txn);
		void writeJobs(job::Job* jobs, int64_t length);
		void writeJobs(job::Job* jobs, int64_t length, Txn* txn);
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
		// void finishJobs_startBuilding();
		// void finishJobs_add(JobID_t job_id);
		// void finishJobs_finishBuilding(Txn* txn);
		void finishJobs(job::Job* jobs, int64_t length);
		void finishJobs(job::Job* jobs, int64_t length, Txn* txn);
		void printJobsStats();
		void printFoundSentence(
			FrequencyMapIndex_t start,
			JobID_t parent_id,
			Dictionary* dict,
			shared_ptr<vector<FrequencyMapIndex_t>> indices,
			Txn* txn
		);
		void printFoundSentences(Dictionary* dict);
	};
}
