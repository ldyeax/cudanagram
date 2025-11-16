#define DICTIONARY_DEBUG 1
#include "definitions.hpp"
#include "dictionary.hpp"
#include "avx.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <cassert>
#include "job.hpp"


using namespace std;
int main()
{
	char test_buffer[] = "abc\0abd\0\0bad\0\0\0cat\ncta\r\ndzz\0a\xee" "b\0";
	int32_t test_buffer_length = sizeof(test_buffer);
	dictionary::Dictionary d = dictionary::Dictionary(
		"abcdt",
		NULL,
		test_buffer,
		test_buffer_length
	);
	d.printStats();

	for (int32_t i = 0; i < d.words.size(); i++) {
		cerr << d.words[i] << " ";
	}
	cerr << endl;

	assert(d.stats.initial_words_parsed == 6);
	assert(d.stats.initial_words_removed == 1);
	assert(d.stats.frequency_map_rejections == 1);
	cerr << "Frequency maps length: " << d.frequency_maps_length << endl;
	cerr << "Passed" << endl;

	char test_buffer_2[] = "helloworld\0hello\0\0world\0\0\0worl\nwor\r\nldhe\0a\xee" "b\0llo\0";
	d = dictionary::Dictionary(
		"helloworld",
		NULL,
		test_buffer_2,
		sizeof(test_buffer_2)
	);
	auto v = d.createInitialjobs(3);
	cerr << "Created " << v->size() << " initial jobs" << endl;
	for (int64_t i = 0; i < v->size(); i++) {
		cerr << "Job " << v->at(i).job_id << ": parent_job_id=" << v->at(i).parent_job_id
			 << " start=" << v->at(i).start
			 << " finished=" << v->at(i).finished << " is_sentence=" << v->at(i).is_sentence
			 << endl;
	}

	// d = dictionary::Dictionary(
	// 	"twomilkmengocomedy",
	// 	"dictionary.txt",
	// 	NULL,
	// 	-1
	// );
	// d.printStats();
}
