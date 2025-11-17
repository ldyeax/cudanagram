/**
 * Stub for builds that don't use GPU
 */
#include "worker.hpp"

worker::WorkerFactory* worker::getWorkerFactory_GPU()
{
	return nullptr;
}

