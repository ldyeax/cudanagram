#include <memory>
#include "database.hpp"
#include <stdio.h>
#include <iostream>
using std::cout;
using job::Job;
int main()
{
	cerr << "Testing db: " << std::endl;
	database::Database mydb = database::Database();
	cerr << "Creating jobs: " << endl;
	job::Job* buffer = new job::Job[4];
	for (int32_t i = 0; i < 4; i++) {
		Job& job = buffer[i];
		buffer[i].parent_job_id = i;
		for (int32_t j = 0; j < NUM_LETTERS_IN_ALPHABET; j++) {
			if (i == j) {
				job.frequency_map[j] = -i;
			}
			else {
				job.frequency_map[j] = j;
			}
		}
		job.start = i * 2;
		//job.parent_frequency_map_index = i * 3;
		job.finished = (i % 2 == 0);
	}

	for (int32_t i = 0; i < 4; i++) {
		Job& j = buffer[i];
		j.print();
	}

	cerr << "Created jobs" << endl;
	cerr << "Writing jobs to database..." << std::endl;
	mydb.writeNewJobsbuffer, 4);
	cerr << "Written jobs to database." << std::endl;
	cerr << "Reading unfinished jobs from database..." << std::endl;
	job::Job* unfinished;
	int32_t num_unfinished = mydb.getUnfinishedJobs(10, unfinished);
	if (num_unfinished == 0) {
		cerr << "No unfinished jobs found." << std::endl;
	}
	else {
		cerr << "Unfinished jobs:" << num_unfinished << std::endl;
		for (int32_t i = 0; i < num_unfinished; i++) {
			Job& j = unfinished[i];
			if (j.job_id == 0) {
				break;
			}
			j.print();
		}
	}
	cerr << "Read unfinished jobs from database." << std::endl;
	for (int32_t i = 0; i <= 4; i++) {
		cerr << "Reading job " << i << " from database..." << std::endl;
		try {
			job::Job j = mydb.getJob(i);
			j.print();
			cerr << "Read job " << i << " from database." << std::endl;
		} catch (const std::exception& e) {
			cerr << "Failed to read job " << i << ": " << e.what() << std::endl;
		}
	}
	std::cerr << "Done" << std::endl;
}
