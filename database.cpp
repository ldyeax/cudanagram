#ifdef CUDANAGRAM_PSQL
	#include "database_psql.cpp"
#else

	#ifdef CUDANAGRAM_MEMORY
		#include "database_memory.cpp"
	#else

		#ifdef CUDANAGRAM_HASHMAP
			#include "database_hashmap.cpp"
		#else

			#ifdef CUDANAGRAM_MMAP
				#include "database_mmap.cpp"
			#else
				#include "database_sqlite.cpp"
			#endif
		#endif
	#endif
#endif
