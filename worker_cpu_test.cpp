// #define NUM_JOBS_PER_BATCH (1024LL*1024LL*1024LL*2LL)

#include <memory>
#include "worker.hpp"
#include "database.hpp"
#include "dictionary.hpp"
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>


void handler(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}


#include "anagrammer.hpp"

using std::vector;
using database::Database;
using worker::Worker;
using dictionary::Dictionary;
using job::Job;
using database::Txn;
using std::cout;
using std::cerr;
using std::endl;

void handle_sigint(int signal)
{
    std::cerr << "\nCaught signal " << signal << ", exiting\n";
    std::quick_exit(1);
}

// also print stack trace on exceptions



int main(int argc, char* argv[])
{
	signal(SIGSEGV, handler);
	signal(SIGABRT, handler);

	cerr << "Worker CPU Test" << endl;
	cerr << "Contents of worker_cpu_test_dictionary.txt:" << endl;
	system("cat worker_cpu_test_dictionary.txt");
	cerr << "------------------------" <<  endl;
	std::signal(SIGINT, handle_sigint);
	string dummy;
	string input = "HELLOWORLD";
	cerr << "Input: " << input << endl;
	// Anagrammer(int64_t p_num_jobs_per_batch, string p_input, string p_dict_filename);
	anagrammer::Anagrammer a(
		1,
		"HELLOWORLD",
		"worker_cpu_test_dictionary.txt"
		//"twomilkmengocomedy"
	);
	cerr << "worker_cpu_test: initialized Anagrammer" << endl;
	a.initWorkers(true, false);
	cerr << "worker_cpu_test: initialized workers, starting run()" << endl;
	a.run();
	cerr << "worker_cpu_test: run() done, printing found sentences:" << endl;
	a.printFoundSentences();
	cerr << "worker_cpu_test: finished" << endl;

	return 0;
}
