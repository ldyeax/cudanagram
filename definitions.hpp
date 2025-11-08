#ifndef __CUDACC__
	#define __device__
	#define __host__
	#define __global__
#endif
#define NUM_LETTERS_IN_ALPHABET 26
#define NO_MATCH frequency_map::no_match
#define INCOMPLETE_MATCH frequency_map::incomplete_match
#define COMPLETE_MATCH frequency_map::complete_match
#define MAX_DICTIONARY_BUFFER 1024*1024*8
