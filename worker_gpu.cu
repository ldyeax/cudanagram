#include "definitions.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.hpp"
#include <thread>

#ifndef WORKER_GPU_BLOCKS
#define WORKER_GPU_BLOCKS 256
#endif

#ifndef WORKER_GPU_THREADS_PER_BLOCK
#define WORKER_GPU_THREADS_PER_BLOCK 1024
#endif

using namespace worker;

using job::Job;

namespace worker_GPU {
    int deviceCount() 
    {
        int ret;
        cudaError_t error = cudaGetDeviceCount(&ret);

        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        std::cout << "Number of CUDA-enabled GPUs: " << ret << std::endl;
        return ret;
    }
    class Worker_GPU : public Worker {
    public:
        Job* h_jobs;
        int32_t h_jobs_length;
        std::thread h_thread;
        int device_id;
        volatile bool ready = false;
        /*
        
        frequency_map::FrequencyMap tmp = {};
		job::Job tmp_job = {};
		tmp_job.parent_job_id = input.job_id;
		FrequencyMapIndex_t start = input.start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			throw;
		}

		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict->h_compareFrequencyMaps_pip(
				&input.frequency_map,
				i,
				&tmp_job.frequency_map
			);
			if (result == INCOMPLETE_MATCH) {
				tmp_job.start = i;
				last_result.new_jobs.push_back(tmp_job);
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				last_result.found_sentences.push_back(tmp_job);
			}
		}
        
        */
        __global__ void kernel(
            Job d_job,
            Dictionary* dict,
            Job* d_new_jobs,
            int32_t* d_num_new_jobs
        ) 
        {
            frequency_map::FrequencyMap tmp = {};
            job::Job tmp_job = {};
            tmp_job.parent_job_id = d_job.job_id;
            FrequencyMapIndex_t start = d_job.start;
            FrequencyMapIndex_t end = dict->frequency_maps_length;
            if (start >= end) {
                throw;
            }
            int32_t num_new_jobs = 0;
            for (FrequencyMapIndex_t i = start; i < end; i++) {
                frequency_map::Result result = dict->d_compareFrequencyMaps_pip(
                    &d_job.frequency_map,
                    i,
                    &tmp_job.frequency_map
                );
                if (result == INCOMPLETE_MATCH) {
                    tmp_job.finished = false;
                }
                else if (result == COMPLETE_MATCH) {
                    tmp_job.finished = true;
                }
                else {
                    continue;
                }
                tmp_job.start = i;
                d_new_jobs[num_new_jobs++] = tmp_job;
            }
            *d_num_new_jobs = num_new_jobs;
        }
        void doJobs() 
        {

        }
        void loop()
        {
            gpuErrChk(cudaSetDevice(device_id));
            
            Dictionary* d_dict;
            gpuErrChk(cudaMalloc(&d_dict, sizeof(Dictionary)));
            gpuErrChk(cudaMemcpy(
                d_dict,
                dict,
                sizeof(Dictionary),
                cudaMemcpyHostToDevice
            ));
            while (true) {
                if (ready) {
                    doJobs();
                    ready = false;
                }
                else {
                    std::this_thread::yield();
                }
            }
        }
        Worker_GPU(int p_device_id) 
        {
            device_id = p_device_id;
            h_thread = std::thread(&Worker_GPU::loop, this);
        }

        int32_t takeJobs(Job* buffer, int32_t max_length) override {
            h_jobs = buffer;
            h_jobs_length = std::min(max_length, h_jobs_length);
            return h_jobs_length;
        }

        void doJobs_async() override {
            // Implementation for asynchronous job processing
        }

        int32_t numThreads() override {
            return WORKER_GPU_BLOCKS * WORKER_GPU_THREADS_PER_BLOCK;
        }
    };
    class WorkerFactory_GPU : public WorkerFactory {
    public:
        int32_t Spawn(
            Worker* buffer,
            int32_t max,
            database::Database* db,
            dictionary::Dictionary* dict
        ) override {
            int32_t num_devices = deviceCount();
            int32_t num_to_spawn = std::min(max, num_devices);
            for (int32_t i = 0; i < num_to_spawn; i++) {
                new (&buffer[i]) Worker_GPU(i);
            }
            return num_to_spawn;
        }
    };

}
