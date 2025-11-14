#include "definitions.hpp"
#include "database.hpp"
#include "job.hpp"
#include <iostream>
#include <cstdlib>
#include <string>

using database::Database;
using namespace std;
using job::Job;

int main()
{
	Database db;
	if (db.getDatabaseType() != DB_TYPE_SQLITE) {
		cerr << "Database type is not SQLITE!" << endl;
		return 1;
	}
	Job test1 = {
		.job_id = 1234,
		.parent_job_id = 9999,
		.frequency_map = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26},
		.start = 256,
		.finished = false,
		.is_sentence = false
	};
	db.setJobIDIncrementStart(2000);
	db.writeJobs(&test1, 1);
	Job tmp[1024];
	int64_t count = db.getUnfinishedJobs(1024, tmp);
	cout << "Unfinished jobs count: " << count << endl;
	if (count != 1) {
		cerr << "Expected 1 unfinished job!" << endl;
		return 1;
	}
	tmp[0].print();
	if (tmp[0].job_id != 2001) {
		cerr << "Expected job ID 2001!" << endl;
		return 1;
	}
	if (tmp[0].parent_job_id != test1.parent_job_id ||
		memcmp(tmp[0].frequency_map.frequencies, test1.frequency_map.frequencies, NUM_LETTERS_IN_ALPHABET) != 0 ||
		tmp[0].start != test1.start ||
		tmp[0].finished != test1.finished ||
		tmp[0].is_sentence != test1.is_sentence
	) {
		cerr << "Fetched job does not match written job!" << endl;
		return 1;
	}
	Database child1 = Database(&db);
	child1.setJobIDIncrementStart(5000);
	Job test2 = {
		.job_id = 0,
		.parent_job_id = 2001,
		.frequency_map = {26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1},
		.start = 512,
		.finished = false,
		.is_sentence = false
	};
	child1.writeJobs(&test2, 1);
	Job tmp2[1024];
	int64_t count2 = child1.getUnfinishedJobs(1024, tmp2);
	cout << "Child unfinished jobs count: " << count2 << endl;
	if (count2 != 1) {
		cerr << "Expected 1 unfinished job in child!" << endl;
		return 1;
	}
	tmp2[0].print();
	if (tmp2[0].job_id != 5001) {
		cerr << "Expected job ID 5001 in child!" << endl;
		return 1;
	}
	count2 = db.getUnfinishedJobs(1024, tmp2);
	cout << "Parent unfinished jobs count after child write: " << count2 << endl;
	if (count2 != 2) {
		cerr << "Expected 2 unfinished jobs in parent after child write!" << endl;
		return 1;
	}
	cout << "tmp2:" << endl;
	tmp2[0].print();
	tmp2[1].print();

	for (int64_t i = 0; i < 1024; i++) {
		tmp[i].job_id = 0;
		tmp[i].parent_job_id = i;
		tmp[i].start = i;
		tmp[i].finished = false;
		tmp[i].is_sentence = false;
	}
	for (int64_t i = 0; i < 1024; i++) {
		tmp2[i].job_id = 0;
		tmp2[i].parent_job_id = i + 10000;
		tmp2[i].start = i + 10000;
		tmp2[i].finished = false;
		tmp2[i].is_sentence = false;
	}

	Job* tmp_output = new Job[4096];
	cout << "Small write test to parent: " << endl;
	db.writeJobs(tmp, 1024);
	int64_t count3 = db.getUnfinishedJobs(2048, tmp_output);
	cout << "Unfinished jobs count after small write: " << count3 << endl;
	if (count3 != 1026) {
		cerr << "Expected 1026 unfinished jobs after small write!" << endl;
		return 1;
	}

	child1.writeJobs(tmp2, 1024);
	cout << "Small write test to child: " << endl;
	int64_t count4 = db.getUnfinishedJobs(4096, tmp_output);
	cout << "Unfinished jobs count after child small write: " << count4 << endl;
	if (count4 != 2050) {
		cerr << "Expected 2050 unfinished jobs after child small write!" << endl;
		return 1;
	}

	cout << "tmp_output[0]:" << endl;
	tmp_output[0].print();
	cout << "tmp_output[1]:" << endl;
	tmp_output[1].print();
	cout << "tmp_output[2]:" << endl;
	tmp_output[2].print();
	cout << "tmp_output[1025]:" << endl;
	tmp_output[1025].print();
	cout << "tmp_output[1026]:" << endl;
	tmp_output[1026].print();
	cout << "tmp_output[1027]:" << endl;
	tmp_output[1027].print();
	cout << "tmp_output[2048]:" << endl;
	tmp_output[2048].print();
	cout << "tmp_output[2049]:" << endl;
	tmp_output[2049].print();

	return 0;
}
