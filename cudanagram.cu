#include <iostream>
#include "anagrammer.hpp"

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

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int64_t worker::max_cpu_threads = -1;
int64_t worker::max_gpu_devices = -1;
bool worker::delay_sentences = false;
atomic<bool> worker::terminated {false};

void handler(int sig) {
//   void *array[10];
//   size_t size;

//   // get void*'s for all entries on the stack
//   size = backtrace(array, 10);

//   // print out all the frames to stderr
//   fprintf(stderr, "Error: signal %d:\n", sig);
//   backtrace_symbols_fd(array, size, STDERR_FILENO);
//   exit(1);
	worker::terminated = true;
}

int printUsage(int argc, char** argv)
{
	cerr << "Usage: "
		<< "\t" << argv[0] << endl
		<< "\t\t[--input] <input_string>" << endl
		<< "\t\t[--dictionary <dictionary_file>]" << endl
		<< "\t\t[--no-cpu]" << endl
		<< "\t\t[--no-gpu]" << endl
		<< "\t\t[--max-cpu-threads <num_threads>]" << endl
		<< "\t\t[--max-gpu-devices <num_devices>]" << endl
		<< "\t\t[--memory-db] (use memory db for all workers)" << endl
		<< "\t\t[--gpu-memory-db] (use memory db for gpu workers)" << endl
		<< "\t\t[--delay-sentences] (delay writing sentences to files)" << endl
		<< "\t\t[--resume]" << endl;
	return 1;
}

int main(int argc, char** argv)
{
	signal(SIGSEGV, handler);
	signal(SIGABRT, handler);
	signal(SIGINT, handler);

	if (argc < 2) {
		return printUsage(argc, argv);
	}
	bool use_cpu = true;
	bool use_gpu = true;
	string input = "";
	//string continue_db = "";
	//bool print_dict = false;
	string dict_filename = "dictionary.txt";
	bool resume = false;
	for (int i = 1; i < argc; i++) {
		string arg(argv[i]);
		cerr << "Arg " << i << ": " << arg << endl;
		if (arg == "--input" && i + 1 < argc) {
			input = argv[i + 1];
			i++;
		}
		else if (arg == "--no-cpu") {
			use_cpu = false;
		}
		else if (arg == "--no-gpu") {
			use_gpu = false;
		}
		// else if (arg == "--continue" && i + 1 < argc) {
		// 	continue_db = argv[i + 1];
		// 	i++;
		// }
		// else if (arg == "--print-dict") {
		// 	print_dict = true;
		// }
		else if (arg == "--dictionary" && i + 1 < argc) {
			dict_filename = argv[i + 1];
			i++;
		}
		else if (arg == "--max-cpu-threads" && i + 1 < argc) {
			worker::max_cpu_threads = std::stoll(argv[i + 1]);
			cerr << "Setting max_cpu_threads = "
				<< worker::max_cpu_threads << endl;
			i++;
		}
		else if (arg == "--max-gpu-devices" && i + 1 < argc) {
			worker::max_gpu_devices = std::stoll(argv[i + 1]);
			cerr << "Setting max_gpu_devices = "
				<< worker::max_gpu_devices << endl;
			i++;
		}
		else if (arg == "--memory-db") {
			database::use_memory_db = true;
		}
		else if (arg == "--gpu-memory-db") {
			database::gpu_memory_db = true;
		}
		else if (arg == "--delay-sentences") {
			worker::delay_sentences = true;
		}
		else if (input == "") {
			input = arg;
		}
		else if (arg == "--resume") {
			resume = true;
			cerr << "Set resume to " << resume << endl;
		}
		else {
			return printUsage(argc, argv);
		}
	}
	anagrammer::Anagrammer a(
		input,
		use_cpu,
		use_gpu,
		dict_filename,
		resume
	);

	return 0;
}
