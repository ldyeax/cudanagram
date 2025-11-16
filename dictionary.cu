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
using job::Job;
#include <mutex>
std::mutex global_dictionary_mutex;

// arbitrarily put this here
std::mutex global_print_mutex;

void dictionary::Dictionary::printStats()
{
	cerr << "Initial words parsed: " << stats.initial_words_parsed << endl;
	cerr << "Initial words removed: " << stats.initial_words_removed << endl;
	cerr << "Frequency map rejections: " << stats.frequency_map_rejections << endl;
	cerr << "Frequency maps length: " << frequency_maps_length << endl;
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
#ifdef DICTIONARY_DEBUG
	cerr << "Dictionary init()" << endl;
#endif
	stats = {};
	words = vector<string>();

	int32_t i_tmp = 0;
	char tmp[256] = {};
	int8_t tmp2[NUM_LETTERS_IN_ALPHABET] = {};
	for (int32_t i = 0; i < buffer_length; i++) {
#ifdef DICTIONARY_DEBUG
		cerr << i << ": " << (int8_t)(buffer[i]) << " " << buffer[i] << endl;
#endif
		if (!buffer[i] || buffer[i] == '\r' || buffer[i] == '\n') {
			i_tmp = 0;
#ifdef DICTIONARY_DEBUG
			cerr << "continuing" << endl;
#endif
			continue;
		}
		if (buffer[i] >= 'a' && buffer[i] <= 'z') {
			buffer[i] = buffer[i] - 'a' + 'A';
		}
		if (buffer[i] < 'A' || buffer[i] > 'Z') {
#ifdef DICTIONARY_DEBUG
			cerr << "Invalid" << endl;
#endif
			stats.initial_words_removed++;
			i_tmp = 0;
			while (i < buffer_length && buffer[i]) {
				i++;
			}
#ifdef DICTIONARY_DEBUG
			cerr << "brought i up to " << i << endl;
#endif
			continue;
		}
		tmp[i_tmp++] = buffer[i];
		if (i_tmp >= 256) {
			throw new std::runtime_error("unspecified");
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
#ifdef DICTIONARY_DEBUG
			cerr << "adding word at " << i << endl;
#endif
			tmp[i_tmp] = 0;
#ifdef DICTIONARY_DEBUG
			fprintf(stderr, "tmp: %s;\n", &tmp[0]);
#endif
			i_tmp = 0;
			stats.initial_words_parsed++;
			FrequencyMap fm = createFrequencyMap(tmp);
#ifdef DICTIONARY_DEBUG
			fm.print();
#endif
			if (avx::compare(
				input_frequency_map,
				fm,
				tmp2
			).any_negative) {
				stats.frequency_map_rejections++;
#ifdef DICTIONARY_DEBUG
				cerr << "rejection: ";
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
	if (frequency_maps_length == 0) {
		cerr << "No frequency maps created in dictionary init()" << endl;
		throw new std::runtime_error("unspecified");
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
		throw new std::runtime_error("unspecified");
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
			throw new std::runtime_error("unspecified");
		}
		buffer = new char[MAX_DICTIONARY_BUFFER];
		buffer_length = fread(
			buffer,
			sizeof(char),
			MAX_DICTIONARY_BUFFER,
			fp
		);
		if (ferror(fp) != 0) {
			throw new std::runtime_error("unspecified");
		}
		fclose(fp);
		init();
	}
	else {
		throw new std::runtime_error("unspecified");
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
#endif

__host__ frequency_map::Result dictionary::Dictionary::h_compareFrequencyMaps_pip(
	frequency_map::FrequencyMap* input,
	FrequencyMapIndex_t other_index,
	frequency_map::FrequencyMap* output
)
{
	frequency_map::FrequencyMap* other = getFrequencyMapPointer(other_index);
    if (other == NULL) {
		throw new std::runtime_error("unspecified");
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
				printf(" ");
			}
			printf(
				"%s",
				words[*(iterators[i])].c_str()
			);
		}
		printf("\n");
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

int64_t processJob(Dictionary* dict, int64_t index, shared_ptr<vector<Job>> existing_jobs)
{
	job::Job tmp_job = {};
	tmp_job.parent_job_id = existing_jobs->at(index).job_id;
	FrequencyMapIndex_t start = existing_jobs->at(index).start;
	FrequencyMapIndex_t end = dict->frequency_maps_length;
	if (start >= end) {
		throw new std::runtime_error("unspecified");
	}
	int64_t ret = 0;
	for (FrequencyMapIndex_t i = start; i < end; i++) {
		frequency_map::Result result = dict->h_compareFrequencyMaps_pip(
			&existing_jobs->at(index).frequency_map,
			i,
			&tmp_job.frequency_map
		);
		if (result == INCOMPLETE_MATCH) {
			cerr << "INCOMPLETE_MATCH found in processJob at " << i << endl;
			tmp_job.job_id = existing_jobs->size() + 1;
			tmp_job.start = i;
			tmp_job.is_sentence = false;
			tmp_job.finished = false;
			existing_jobs->push_back(tmp_job);
			ret++;
		}
		else if (result == COMPLETE_MATCH) {
			cerr << "COMPLETE_MATCH found in processJob at " << i << endl;
			tmp_job.job_id = existing_jobs->size() + 1;
			tmp_job.start = i;
			tmp_job.is_sentence = true;
			tmp_job.finished = true;
			existing_jobs->push_back(tmp_job);
			ret++;
		}
	}

	cerr << "Marking job " << existing_jobs->at(index).job_id << " as finished" << endl;
	existing_jobs->at(index).finished = true;

	return ret;
}

shared_ptr<vector<Job>> dictionary::Dictionary::createInitialjobs(int64_t count)
{
	#ifdef DICTIONARY_DEBUG
	count = 1;
	#endif
	std::lock_guard<std::mutex> lock(global_dictionary_mutex);
	cerr << "Dictionary: Creating initial jobs up to count " << count << endl;
	cerr << "frequency maps length = " << frequency_maps_length << endl;
	shared_ptr<vector<Job>> ret = make_shared<vector<Job>>();
	int64_t to_reserve = count * frequency_maps_length * 2L;
	if (to_reserve < 0) {
		cerr << "bad to_reserve calculation: " << to_reserve << endl;
		throw new std::runtime_error("unspecified");
	}
	if (to_reserve > 65535) {
		to_reserve = 65535;
	}
	cerr << "Reserving space for up to " << to_reserve << " jobs" << endl;
	ret->reserve(to_reserve);
	//ret->push_back(start_job);
	Job& start_job = ret->emplace_back();
	start_job.job_id = 1;
	start_job.parent_job_id = 0;
	copyInputFrequencyMap(&start_job.frequency_map);
	start_job.start = 0;
	start_job.finished = false;
	start_job.is_sentence = false;
	start_job.finished = false;
	if (start_job.frequency_map.isAllZero()) {
		cerr << "createInitialjobs: all-zero frequency map in start job" << endl;
		throw new std::runtime_error("unspecified");
	}
	if (start_job.frequency_map.anyNegative()) {
		cerr << "createInitialjobs: negative frequency map value in start job" << endl;
		throw new std::runtime_error("unspecified");
	}
	int64_t unfinished_count = 0;
	int64_t finished_count = 0;
	while (unfinished_count < count) {
		int64_t initial_size = ret->size();
		int64_t added = 0;
		for (int32_t i = 0; i < initial_size; i++) {
			if (!ret->at(i).finished) {
				cerr << "createInitialjobs: processing job " << ret->at(i).job_id << " unfinished_count=" << unfinished_count << endl;
				added += processJob(this, i, ret);
			}
		}
		if (added == 0) {
			cerr << "createInitialjobs: no more jobs can be created, breaking at unfinished_count=" << unfinished_count << endl;
			break;
		}
		unfinished_count = 0;
		for (int64_t i = 0; i < ret->size(); i++) {
			if (!ret->at(i).finished) {
				unfinished_count++;
			}
		}
	}
	cerr << "Created " << ret->size() << " initial jobs (" << unfinished_count << " unfinished, " << (ret->size() - unfinished_count) << " finished)" << endl;
	for (int64_t i = 0; i < ret->size(); i++) {
		if (!ret->at(i).finished) {
			if (ret->at(i).frequency_map.isAllZero()) {
				cerr << "createInitialjobs: all-zero frequency map in job " << ret->at(i).job_id << endl;
				throw new std::runtime_error("all-zero frequency map in job");
			}
			if (ret->at(i).frequency_map.anyNegative()) {
				cerr << "createInitialjobs: negative frequency map value in job " << ret->at(i).job_id << endl;
				throw new std::runtime_error("negative frequency map value in job");
			}
		}
	}

	Job& start_job_2 = ret->at(0);
	start_job_2.finished = true;

	int64_t unfinished_count_2 = 0;
	for (int64_t i = 0; i < ret->size(); i++) {
		if (!ret->at(i).finished) {
			unfinished_count_2++;
		}
	}
	// cerr << "start_job_2.finished set to true" << endl;
	cerr << "unfinished_count_2: " << unfinished_count_2 << endl;

	// cerr << "start_job_2.finished = true" << endl;
	// cerr << "start_job_2.finished: " << start_job_2.finished << endl;
	// cerr << "ret[0].finished: " << ret->at(0).finished << endl;
	return ret;
}

void dictionary::Dictionary::printWordsAt(FrequencyMapIndex_t fm_index)
{
	if (fm_index < 0 || fm_index >= frequency_maps_length) {
		cerr << "Invalid frequency map index: " << fm_index << endl;
		return;
	}
	for (auto w_i : word_index_lists_for_frequency_maps[fm_index]) {
		cout << words[w_i] << endl;
	}
}
void dictionary::Dictionary::printWordsAt(FrequencyMapIndex_t fm_index, int32_t depth)
{
	if (fm_index < 0 || fm_index >= frequency_maps_length) {
		cerr << "Invalid frequency map index: " << fm_index << endl;
		return;
	}
	for (auto w_i : word_index_lists_for_frequency_maps[fm_index]) {
		if (depth-- <= 0) {
			return;
		}
		cout << words[w_i] << endl;
	}
}

void dictionary::Dictionary::printDict()
{
	for (FrequencyMapIndex_t i = 0; i < frequency_maps_length; i++) {
		printWordsAt(i);
	}
}
