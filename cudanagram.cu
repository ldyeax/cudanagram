#include <iostream>
#include "anagrammer.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int printUsage(int argc, char** argv)
{
	cerr << "Usage: " << argv[0] <<
		" [--input] <input_string> "
		"[--dictionary <dictionary_file>] "
		"[--continue <database_name>] "
		"[--jobs-per-batch <num>] "
		"[--no-cpu] [--no-gpu] "
		"" << std::endl;
	return 1;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		return printUsage(argc, argv);
	}
	bool use_cpu = true;
	bool use_gpu = true;
	string input = "";
	string continue_db = "";
	int64_t num_jobs_per_batch = 1073741824LL; // 161064LL;
	bool print_dict = false;
	string dict_filename = "dictionary.txt";
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
		else if (arg == "--jobs-per-batch" && i + 1 < argc) {
			num_jobs_per_batch = atoll(argv[i + 1]);
			i++;
		}
		else if (arg == "--continue" && i + 1 < argc) {
			continue_db = argv[i + 1];
			i++;
		}
		else if (arg == "--print-dict") {
			print_dict = true;
		}
		else if (arg == "--dictionary" && i + 1 < argc) {
			dict_filename = argv[i + 1];
			i++;
		}
		else if (input == "") {
			input = arg;
		}
		else {
			return printUsage(argc, argv);
		}
	}
	anagrammer::Anagrammer a(
		num_jobs_per_batch,
		input,
		dict_filename
	);
	if (print_dict) {
		a.printDict();
		return 0;
	}
	a.initWorkers(use_cpu, use_gpu);
	a.run();
	a.printFoundSentences();
	return 0;
}
