/**
 * Stub for builds that don't use GPU
 */
#include "worker.hpp"
#include "database.hpp"
#include "dictionary.hpp"
worker::WorkerFactory* worker::getWorkerFactory_GPU(database::Database* db, dictionary::Dictionary* dict)
{
	return nullptr;
}

