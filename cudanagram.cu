#include <iostream>
#include "anagrammer.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int printUsage(int argc, char** argv)
{
	cerr << "Usage: " << argv[0] << " <input string> [--jobs-per-batch <num>] [--no-cpu] [--no-gpu]" << std::endl;
	return 1;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		return printUsage(argc, argv);
	}
	bool use_cpu = true;
	bool use_gpu = true;
	string input(argv[1]);
	int64_t num_jobs_per_batch = 1073741824LL; // 161064LL;
	for (int i = 2; i < argc; i++) {
		string arg(argv[i]);
		if (arg == "--no-cpu") {
			use_cpu = false;
		} else if (arg == "--no-gpu") {
			use_gpu = false;
		} else if (arg == "--jobs-per-batch" && i + 1 < argc) {
			num_jobs_per_batch = atoll(argv[i + 1]);
			i++;
		} else {
			return printUsage(argc, argv);
		}
	}
	anagrammer::Anagrammer a(
		num_jobs_per_batch,
		input
	);
	a.initWorkers(use_cpu, use_gpu);
	a.run();
	a.printFoundSentences();
	return 0;
}
