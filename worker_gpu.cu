#define DTEST_WORKER_GPU 1

#include "definitions.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.cuh"
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
	__global__ void kernel(
		Job* d_job,
		Dictionary* dict,
		Job* d_new_jobs,
		/**
		 * Array such that d_num_new_jobs[i] is the number of new jobs spawned by input job i
		 */
		int64_t* d_num_new_jobs,
		int64_t num_input_jobs
	)
	{
		int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= num_input_jobs) {
			return;
		}
		d_job += index;
		job::Job tmp_job = {};
		tmp_job.parent_job_id = d_job->job_id;
		FrequencyMapIndex_t start = d_job->start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			printf("failure\n");
			return;
		}
		d_new_jobs += index * end;
		int32_t num_new_jobs = 0;
		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict->d_compareFrequencyMaps_pip(
				&d_job->frequency_map,
				i,
				&tmp_job.frequency_map
			);
			if (result == NO_MATCH) {
				continue;
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.is_sentence = true;
			}
			tmp_job.start = i;
			d_new_jobs[num_new_jobs++] = tmp_job;
		}
		d_num_new_jobs[index] = num_new_jobs;
	}

    int deviceCount()
    {
        int ret;
        cudaError_t error = cudaGetDeviceCount(&ret);

        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        std::cerr << "Number of CUDA-enabled GPUs: " << ret << std::endl;
        return ret;
    }
    class Worker_GPU : public Worker {
private:
		Dictionary* d_dict = nullptr;
		Job* d_input_jobs = nullptr;
		Job* d_new_jobs = nullptr;
		Job* h_new_jobs_tmp = nullptr;

		int64_t* d_num_new_jobs = nullptr;
		int64_t* h_num_new_jobs = nullptr;

		int64_t max_new_jobs_per_job;
		int64_t max_input_jobs_per_iteration;

		Job* h_input_jobs = nullptr;
public:
        vector<Job*> h_unfinished_jobs;
        std::thread h_thread;
        int device_id;

		void reset() override {
			Worker::reset();
			h_unfinished_jobs.clear();
		}

		void doJob(Job* d_input_jobs, int64_t p_count) override
		{
			#ifdef DTEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: processing %ld jobs on device %d\n", p_count, device_id);
			fprintf(stderr, "Worker_GPU::doJob: max_input_jobs_per_iteration=%ld\n", max_input_jobs_per_iteration);
			#endif
			// launch kernel
			dim3 blocks(WORKER_GPU_BLOCKS);
			dim3 threads(WORKER_GPU_THREADS_PER_BLOCK);
			#ifdef DTEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: launching kernel with %d blocks of %d threads\n", blocks.x, threads.x);
			#endif
			kernel<<<blocks, threads>>>(
				d_input_jobs,
				d_dict,
				d_new_jobs,
				d_num_new_jobs,
				p_count
			);
			gpuErrChk(cudaDeviceSynchronize());
			// copy number of new jobs back to host
			gpuErrChk(cudaMemcpy(
				h_num_new_jobs,
				d_num_new_jobs,
				sizeof(int64_t) * p_count,
				cudaMemcpyDeviceToHost
			));
			// copy whole new jobs buffer back to host
			int64_t num_total_new_jobs = 0;
			for (int64_t i = 0; i < max_input_jobs_per_iteration; i++) {
				int64_t num_new_jobs_i = h_num_new_jobs[i];
				#ifdef DTEST_WORKER_GPU
				fprintf(stderr, "Worker_GPU::doJob: job %ld produced %ld new jobs\n", i, num_new_jobs_i);
				// read line to pause
				std::string dummy;
				std::getline(std::cin, dummy);
				#endif
				num_total_new_jobs += num_new_jobs_i;
				Job* tmp = h_new_jobs_tmp + (i * max_new_jobs_per_job);
				for (int64_t j = 0; j < num_new_jobs_i; j++) {
					last_result.new_jobs.push_back(*tmp);
					tmp++;
				}
			}
		}

		void doJobs()
        {
			int64_t jobs_done = 0;
			int64_t num_unfinished_jobs = h_unfinished_jobs.size();

			last_result.new_jobs.clear();

			while (jobs_done < num_unfinished_jobs) {
				memset(h_num_new_jobs, 0, sizeof(int64_t) * max_input_jobs_per_iteration);

				int64_t jobs_start = jobs_done;
				int64_t jobs_end = jobs_start + max_input_jobs_per_iteration;
				if (jobs_end > num_unfinished_jobs) {
					jobs_end = num_unfinished_jobs;
				}
				int64_t num_input_jobs = jobs_end - jobs_start;
				for (int64_t i = 0; i < num_input_jobs; i++) {
					h_input_jobs[i] = *(h_unfinished_jobs[jobs_start + i]);
				}
				// copy input jobs to device
				gpuErrChk(cudaMemcpy(
					d_input_jobs,
					h_input_jobs,
					sizeof(Job) * num_input_jobs,
					cudaMemcpyHostToDevice
				));
				// process each job
				doJob(
					d_input_jobs,
					num_input_jobs
				);
				jobs_done += num_input_jobs;
			}
        }
        void loop() override
        {
			Database thread_db = Database(db);

			max_input_jobs_per_iteration = numThreads();
			max_new_jobs_per_job = dict->frequency_maps_length;

			h_input_jobs = new Job[max_input_jobs_per_iteration];
			h_new_jobs_tmp = new Job[max_new_jobs_per_job * max_input_jobs_per_iteration];
			h_num_new_jobs = new int64_t[max_input_jobs_per_iteration];

            gpuErrChk(cudaSetDevice(device_id));
            gpuErrChk(cudaMalloc(&d_dict, sizeof(Dictionary)));
            gpuErrChk(cudaMemcpy(
                d_dict,
                dict,
                sizeof(Dictionary),
                cudaMemcpyHostToDevice
            ));

			gpuErrChk(cudaMalloc(
				&d_input_jobs,
				sizeof(Job) * max_input_jobs_per_iteration
			));

			gpuErrChk(cudaMalloc(
				&d_new_jobs,
				sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
			));

			gpuErrChk(cudaMalloc(
				&d_num_new_jobs,
				sizeof(int32_t)* max_new_jobs_per_job * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaMemset(
				d_num_new_jobs,
				0,
				sizeof(int32_t)
			));

			gpuErrChk(cudaDeviceSynchronize());

            while (true) {
                if (ready_to_start) {
					ready_to_start = false;
                    doJobs();

					auto txn = thread_db.beginTransaction();
					WriteResult(&last_result, dict, txn);
					thread_db.commitTransaction(txn);
					finished = true;
                }
                else {
                    //std::this_thread::yield();
					std::this_thread::sleep_for(std::chrono::microseconds(250000));
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
				(int64_t)numThreads(), max_length
			);
            for (int64_t i = 0; i < to_take; i++) {
				h_unfinished_jobs.push_back(buffer++);
			}
            return to_take;
        }

        void doJobs_async() override {
            ready_to_start = true;
        }

        int32_t numThreads() override {
            return WORKER_GPU_BLOCKS * WORKER_GPU_THREADS_PER_BLOCK;
        }
    };
    class WorkerFactory_GPU : public WorkerFactory {
    public:
        int64_t Spawn(
            Worker** buffer,
            int64_t max,
            database::Database* db,
            dictionary::Dictionary* dict
        ) override {
            int32_t num_devices = deviceCount();
            int64_t num_to_spawn
				= std::min(max, (int64_t)num_devices);
            for (int64_t i = 0; i < num_to_spawn; i++) {
                buffer[i] = new Worker_GPU(
					db,
					dict,
					i
				);
            }
            return num_to_spawn;
        }
    };
}


WorkerFactory* worker::getWorkerFactory_GPU(database::Database* db, dictionary::Dictionary* dict)
{
	static WorkerFactory* ret = new worker_GPU::WorkerFactory_GPU();
	return ret;
}
