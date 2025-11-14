#pragma once
#include <cstdint>
#define databaseType_t int8_t
#define DB_TYPE_PSQL 1
#define DB_TYPE_SQLITE 2
#define DB_TYPE_HASHMAP 3
#define DB_TYPE_MMAP 4

#include <cstdio>
#ifdef __CUDACC__
	/**
	* Wrappers for cuda calls to check for errors
	**/
	#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

	#define gpuErrChkF(ans, f) { \
			gpuAssert((ans), __FILE__, __LINE__, true, (f)); \
	}

	/**
	 * If code is not success:
	 * - run optional beforePrint function
	 * - print error, file, line
	 * - exit program if abort=true
	 **/
	inline void gpuAssert(
			cudaError_t code,
			const char *file,
			int line,
			bool abort = true,
			void (*beforePrint)() = NULL)
	{
		// fprintf(stderr, "%s\n", cudaGetErrorString(code));
		if (code != cudaSuccess) {
			if (beforePrint) {
				beforePrint();
			}
			fprintf(stderr,
			// ffprintf(stderr,
			//	stderr,
				"GPUassert: %s %s %d\n",
				cudaGetErrorString(code),
				file,
				line);
			if (abort) {
				fprintf(stderr, "abort");
				exit(code);
			}
		}
	}
#else
	#define __device__
	#define __host__
	#define __global__
#endif
#define NUM_LETTERS_IN_ALPHABET 26
#define NO_MATCH frequency_map::no_match
#define INCOMPLETE_MATCH frequency_map::incomplete_match
#define COMPLETE_MATCH frequency_map::complete_match
#define MAX_DICTIONARY_BUFFER 1024*1024*8
