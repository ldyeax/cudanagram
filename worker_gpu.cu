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
        vector<Job*> h_unfinished_jobs;
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
			for (Job* job_ptr : h_unfinished_jobs) {
				Job& job = *job_ptr;
				// copy job to device
				Job d_job;
				gpuErrChk(cudaMemcpy(
					&d_job,
					&job,
					sizeof(Job),
					cudaMemcpyHostToDevice
				));
				// allocate device memory for new jobs
				Job* d_new_jobs;
				int32_t max_new_jobs = dict->frequency_maps_length - job.start;
				gpuErrChk(cudaMalloc(&d_new_jobs, sizeof(Job) * max_new_jobs));
				int32_t* d_num_new_jobs;
				gpuErrChk(cudaMalloc(&d_num_new_jobs, sizeof(int32_t)));
				// launch kernel
				dim3 blocks(WORKER_GPU_BLOCKS);
				dim3 threads(WORKER_GPU_THREADS_PER_BLOCK);
				kernel<<<blocks, threads>>>(
					d_job,
					dict,
					d_new_jobs,
					d_num_new_jobs
				);
				gpuErrChk(cudaDeviceSynchronize());
				// copy back number of new jobs
				int32_t h_num_new_jobs = 0;
				gpuErrChk(cudaMemcpy(
					&h_num_new_jobs,
					d_num_new_jobs,
					sizeof(int32_t),
					cudaMemcpyDeviceToHost
				));
				// copy back new jobs
				vector<Job> h_new_jobs(h_num_new_jobs);
				gpuErrChk(cudaMemcpy(
					h_new_jobs.data(),
					d_new_jobs,
					sizeof(Job) * h_num_new_jobs,
					cudaMemcpyDeviceToHost
				));
				// free device memory
				gpuErrChk(cudaFree(d_new_jobs));
				gpuErrChk(cudaFree(d_num_new_jobs));
				// process new jobs
				for (const Job& new_job : h_new_jobs) {
					if (new_job.finished) {
						last_result.found_sentences.push_back(new_job);
					}
					else {
						last_result.new_jobs.push_back(new_job);
					}
				}
			}
			h_unfinished_jobs.clear();
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
					finished = true;
                }
                else {
                    std::this_thread::yield();
                }
            }
        }
        Worker_GPU(
			Database* p_db, Dictionary* p_dict, int p_device_id
		) : Worker(p_db, p_dict)
        {
            device_id = p_device_id;
            h_thread = std::thread(&Worker_GPU::loop, this);
        }

        int64_t takeJobs(Job* buffer, int64_t max_length) override {
			int64_t to_take = std::min(
				numThreads(), max_length
			);
            for (int64_t i = 0; i < to_take; i++) {
				h_unfinished_jobs.push_back(buffer++);
			}
            return to_take;
        }

        void doJobs_async() override {
            ready = true;
        }

        int32_t numThreads() override {
            return WORKER_GPU_BLOCKS * WORKER_GPU_THREADS_PER_BLOCK;
        }
    };
    class WorkerFactory_GPU : public WorkerFactory {
    public:
        int32_t Spawn(
            Worker** buffer,
            int32_t max,
            database::Database* db,
            dictionary::Dictionary* dict
        ) override {
            int32_t num_devices = deviceCount();
            int32_t num_to_spawn = std::min(max, num_devices);
            for (int32_t i = 0; i < num_to_spawn; i++) {
                buffer[i] = new Worker_GPU(db, dict, i);
            }
            return num_to_spawn;
        }
    };

}
