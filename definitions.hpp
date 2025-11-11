#pragma once
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
		// printf("%s\n", cudaGetErrorString(code));
		if (code != cudaSuccess) {
			if (beforePrint) {
				beforePrint();
			}
			printf(
			// fprintf(
			//	stderr, 
				"GPUassert: %s %s %d\n",
				cudaGetErrorString(code),
				file,
				line);
			printf("%-10s\t%p\n", "d_Input",  d_Input);
			printf("%-10s\t%p\n", "d_Output", d_Output);
			printf("%-10s\t%p\n", "h_Input",  h_Input);
			printf("%-10s\t%p\n", "h_Output", h_Output);
			if (abort) {
				printf("abort");
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
