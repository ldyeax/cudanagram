#include "database.hpp"
#include <stdio.h>
#include <iostream>
using std::cout;
using job::Job;
int main()
{
	cout << "Testing db: " << std::endl;
	database::Database mydb = database::Database();
	cout << "Creating jobs: " << endl;
	job::Job* buffer = new job::Job[4];
	for (int32_t i = 0; i < 4; i++) {
		buffer[i].parent_job_id = i;
		for (int32_t j = 0; j < NUM_LETTERS_IN_ALPHABET; j++) {
			if (i == j) {
				job.frequency_map[j] = -i;
			}
			else {
				job.frequency_map[j] = j;
			}
		}
	}
	job.start = i * 2;
	job.finished = (i % 2 == 0);
	for (int32_t i = 0; i < 4; i++) {
		Job& j = buffer[i];
		j.print();
	}
	std::cout << "Done" << std::endl;
}
