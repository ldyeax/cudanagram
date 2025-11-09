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
		cout << d.words[i] << " ";
	}
	cout << endl;

	assert(d.stats.initial_words_parsed == 6);
	assert(d.stats.initial_words_removed == 1);
	assert(d.stats.frequency_map_rejections == 1);
	cout << "Frequency maps length: " << d.frequency_maps_length << endl;
	cout << "Passed" << endl;

	d = dictionary::Dictionary(
		"twomilkmengocomedy",
		"dictionary.txt",
		NULL,
		-1
	);
	d.printStats();
}
