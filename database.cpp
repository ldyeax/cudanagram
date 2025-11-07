#include "database.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <pqxx/pqxx>

using namespace database;

std::string Database::getNewDatabaseName()
{
	auto current_unix_timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
	return current_unix_timestamp;
}

Database::Database(std::string existing_db_name)
{
	throw;
}

Database::Database()
{

	putenv("PGPASSWORD=cudanagram");
	string db_name = getNewDatabaseName();
	std::cout << "db_name = " << db_name << std::endl;
	system("psql -U cudanagram -v dbname=" + db_name + " -f setup.sql");
	pqxx::connection c{"dbname=" + db_name + " user=cudanagram host=/var/run/postgresql"};
	pqxx::work txn{c};
	int8_t test_fm[NUM_LETTERS_IN_ALPHABET] = {1, 2, 3, 0};
	pqxx::binarystring freq_bin(test_fm, sizeof(test_fm));
	pqxx::result r = txn.exec_params(
		"INSERT INTO job (parent_job_id, frequency_map, start) "
		"VALUES ($1, $2, $3) "
		"RETURNING job_id;",
		pqxx::null(),
		freq_bin,
		7
	);
	txn.commit();
	if (r.empty()) {
		throw std::runtime_error("no id returned");
	}
	int64_t job_id = r[0][0].as<int64_t>();
	std::cout << "Inserted job id = " << job_id << std::endl;

}


void Database::writeJob(job::Job job)
{

}

job::Job Database::getJob(JobID_t job_id)
{

}
