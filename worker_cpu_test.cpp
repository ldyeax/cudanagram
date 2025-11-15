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
    std::cerr << "\nCaught Ctrl-C (signal " << signal << "), exiting\n";
    std::quick_exit(1);
}

int main(int argc, char* argv[])
{
	std::signal(SIGINT, handle_sigint);
	string dummy;

	anagrammer::Anagrammer a(
		(1024LL*1024LL),
		"twomilkmengocomedy"
	);
	a.initWorkers(true, false);
	a.run();
	a.printFoundSentences();

	return 0;
}
