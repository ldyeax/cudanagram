#pragma once
#include <cstring>
#include <stdio.h>
#include "job.hpp"
#include <string>
#include <memory>
using std::unique_ptr;
namespace database {
	struct Impl;
	class Database {
	private:
		std::string getNewDatabaseName();
		std::string db_name;
		Impl* impl;
		void create_db();
		void connect();
		void init();
	public:
		~Database();
		Database();
		Database(std::string existing_db_name);
		// Write job to db, where its ID will be created
		void writeJob(job::Job);
		void writeJobs(job::Job* jobs, int32_t length);
		/**
		 * Input should have start=index of last word in sentence
		 * Sentence will be found by traversing parent_job_id
		 * **/
		void writeCompleteSentence(job::Job);
		job::Job getJob(JobID_t job_id);
		job::Job* getUnfinishedJobs(int32_t length);
	};
}
