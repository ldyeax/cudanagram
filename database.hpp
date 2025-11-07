#pragma once
#include <cstring>
#include <stdio.h>
#include "job.hpp"
#include <string>

namespace database {
	class Database {
	private:
		std::string getNewDatabaseName();
	public:
		Database();
		Database(std::string existing_db_name);
		void writeJob(job::Job);
		job::Job getJob(JobID_t job_id);
	};
}
