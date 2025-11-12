#include "definitions.hpp"
#include "dictionary.hpp"
#include "avx.hpp"
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include "job.hpp"
#include <iostream>
using std::shared_ptr;
using std::make_shared;
using namespace dictionary;
using namespace frequency_map;
using std::string;
using std::vector;
using std::endl;
using std::cerr;

void dictionary::Dictionary::printStats()
{
	cerr << "Initial words parsed: " << stats.initial_words_parsed << endl;
	cerr << "Initial words removed: " << stats.initial_words_removed << endl;
	cerr << "Frequency map rejections: " << stats.frequency_map_rejections << endl;
}

FrequencyMapIndex_t dictionary::Dictionary::getOrCreateFrequencyMapIndexByWordIndex(
	int32_t w_i
)
{
	FrequencyMap fm_w = createFrequencyMap(words[w_i]);
	for (FrequencyMapIndex_t i = 0; i < frequency_maps_length; i++) {
		if (avx::compare(fm_w, frequency_maps[i]).all_zero) {
			return i;
		}
	}
	FrequencyMapIndex_t ret = frequency_maps_length;
	frequency_maps_length++;
	frequency_maps[ret] = fm_w;
	return ret;
}

void dictionary::Dictionary::init()
{
#if DICTIONARY_DEBUG
	cerr << "Dictionary init()" << endl;
#endif
	stats = {};
	words = vector<string>();

	int32_t i_tmp = 0;
	char tmp[256] = {};
	int8_t tmp2[NUM_LETTERS_IN_ALPHABET] = {};
	for (int32_t i = 0; i < buffer_length; i++) {
#if DICTIONARY_DEBUG
		cerr << i << ": " << (int8_t)(buffer[i]) << " " << buffer[i] << endl;
#endif
		if (!buffer[i] || buffer[i] == '\r' || buffer[i] == '\n') {
			i_tmp = 0;
#if DICTIONARY_DEBUG
			cerr << "continuing" << endl;
#endif
			continue;
		}
		if (buffer[i] >= 'a' && buffer[i] <= 'z') {
			buffer[i] = buffer[i] - 'a' + 'A';
		}
		if (buffer[i] < 'A' || buffer[i] > 'Z') {
#if DICTIONARY_DEBUG
			cerr << "Invalid" << endl;
#endif
			stats.initial_words_removed++;
			i_tmp = 0;
			while (i < buffer_length && buffer[i]) {
				i++;
			}
#if DICTIONARY_DEBUG
			cerr << "brought i up to " << i << endl;
#endif
			continue;
		}
		tmp[i_tmp++] = buffer[i];
		if (i_tmp >= 256) {
			throw;
		}
		if (
			(i + 1 < buffer_length &&
				(!buffer[i + 1]
				 || buffer[i + 1] == '\r'
				 || buffer[i + 1] == '\n'
				)
			)
			|| i + 1 >= buffer_length
		) {
#if DICTIONARY_DEBUG
			cerr << "adding word at " << i << endl;
#endif
			tmp[i_tmp] = 0;
#if DICTIONARY_DEBUG
			fprintf(stderr, "tmp: %s;\n", &tmp[0]);
#endif
			i_tmp = 0;
			stats.initial_words_parsed++;
			FrequencyMap fm = createFrequencyMap(tmp);
#if DICTIONARY_DEBUG
			fm.print();
#endif
			if (avx::compare(
				input_frequency_map,
				fm,
				tmp2
			).any_negative) {
				stats.frequency_map_rejections++;
#if DICTIONARY_DEBUG
				reinterpret_cast<FrequencyMap*>(tmp2)->print();
#endif
				continue;
			}
			string s = string(tmp);
			words.push_back(s);
		}
	}
	frequency_maps = new FrequencyMap[words.size()];
	word_index_lists_for_frequency_maps = new vector<int32_t>[words.size()]{};
	for (int32_t i = 0; i < words.size(); i++) {
		FrequencyMapIndex_t fm_i = getOrCreateFrequencyMapIndexByWordIndex(
			i
		);
		word_index_lists_for_frequency_maps[fm_i].push_back(i);
	}
}

dictionary::Dictionary::Dictionary(
	char* p_input,
	char* p_filename,
	char* p_buffer,
	int32_t p_buffer_length
)
{
	if (p_input == NULL) {
		throw;
	}
	input = string(p_input);
	input_frequency_map = createFrequencyMap(p_input);
	if (p_filename == NULL && p_buffer != NULL && p_buffer_length > 0) {
		cerr << "Initializing dictionary with buffer" << endl;
		buffer = new char[p_buffer_length];
		buffer_length = p_buffer_length;
		for (int i = 0; i < p_buffer_length; i++) {
			buffer[i] = p_buffer[i];
		}
		init();
	}
	else if (p_filename != NULL && p_buffer == NULL) {
		cerr << "Initializing dictionary with filename" << endl;
		FILE* fp = fopen(p_filename, "r");
		if (fp == NULL) {
			throw;
		}
		buffer = new char[MAX_DICTIONARY_BUFFER];
		buffer_length = fread(
			buffer,
			sizeof(char),
			MAX_DICTIONARY_BUFFER,
			fp
		);
		if (ferror(fp) != 0) {
			throw;
		}
		fclose(fp);
		init();
	}
	else {
		throw;
	}
}

#ifdef __CUDACC__
__device__ frequency_map::Result dictionary::Dictionary::d_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
	if (other == NULL) {
		printf("other_index %d found NULL getFrequencyMapPointer", other_index);
		__trap();
	}
	frequency_map::Result ret = COMPLETE_MATCH;
	for (int8_t i = 0; i < NUM_LETTERS_IN_ALPHABET; i++) {
		int8_t r = ((*output)[i] = (*input)[i] - (*other)[i]);
		if (r < 0) {
			return NO_MATCH;
		}
		else if (r > 0) {
			ret = INCOMPLETE_MATCH;
		}
	}
	return ret;
}
#else
compilation error
#endif

__host__ frequency_map::Result dictionary::Dictionary::h_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
    if (other == NULL) {
		throw;
	}
	auto ret = avx::compare(
		*input,
		*other,
		(int8_t*)output
	);
	if (ret.any_negative) {
		return NO_MATCH;
	}
	return !ret.all_zero ? INCOMPLETE_MATCH : COMPLETE_MATCH;
}

frequency_map::FrequencyMap* dictionary::Dictionary::getFrequencyMapPointer(
	FrequencyMapIndex_t index
)
{
	if (index < 0) {
		return nullptr;
	}
	if (index >= frequency_maps_length) {
		return nullptr;
	}
	return frequency_maps + index;
}

void dictionary::Dictionary::copyInputFrequencyMap(frequency_map::FrequencyMap* dest)
{
	memcpy(dest, &input_frequency_map, NUM_LETTERS_IN_ALPHABET);
}

void dictionary::Dictionary::printSentence(
	shared_ptr<vector<FrequencyMapIndex_t>> p_indices)
{
	auto indices = *p_indices;
	// for each index, get an iterator from word_index_lists_for_frequency_maps
	auto iterators = vector<vector<int32_t>::iterator>();
	for (auto index : indices) {
		iterators.push_back(
			word_index_lists_for_frequency_maps[index].begin()
		);
	}
	while (true) {
		// print current words
		for (int32_t i = 0; i < iterators.size(); i++) {
			if (i > 0) {
				fprintf(stderr, " ");
			}
			fprintf(stderr,
				"%s",
				words[*(iterators[i])].c_str()
			);
		}
		fprintf(stderr, "\n");
		// increment iterators
		int32_t carry = 1;
		for (int32_t i = iterators.size() - 1; i >= 0; i--) {
			if (carry == 0) {
				break;
			}
			carry = 0;
			iterators[i]++;
			if (iterators[i] == word_index_lists_for_frequency_maps[indices[i]].end()) {
				if (i == 0) {
					return;
				}
				iterators[i] = word_index_lists_for_frequency_maps[indices[i]].begin();
				carry = 1;
			}
		}
		// if no iterators advanced (ie we are done), break
		if (carry == 1) {
			break;
		}
	}
}

int32_t dictionary::Dictionary::createInitialjobs(job::Job* buffer)
{
	for (FrequencyMapIndex_t i = 0; i < frequency_maps_length; i++) {
		buffer->parent_job_id = 0;
		copyInputFrequencyMap(&buffer->frequency_map);
		buffer->start = i;
		buffer->finished = false;
		buffer->is_sentence = false;
		buffer++;
	}
	return frequency_maps_length;
}
