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


int main(int argc, char* argv[])
{
	signal(SIGSEGV, handler);
	signal(SIGABRT, handler);

	cerr << "Worker CPU Test" << endl;

	string dummy;
	string input = "HELLOWORLD";
	//string input = "twomilkmengocomedy";
	cerr << "Input: " << input << endl;
	anagrammer::Anagrammer a(
		input,
		true,
		false,
		"worker_cpu_test_dictionary.txt"
	);

	return 0;
}
