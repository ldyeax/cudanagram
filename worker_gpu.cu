//#define TEST_WORKER_GPU 1

#include "definitions.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"
#include "dictionary.cuh"
#include <thread>

using namespace worker;

using job::Job;
using std::cerr;
using std::endl;

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
#ifdef TEST_WORKER_GPU
		if (index == 0) {
			printf("kernel launched with %d blocks of %d threads\n", gridDim.x, blockDim.x);
		}
#endif
		if (index >= num_input_jobs) {
// #ifdef TEST_WORKER_GPU
// 			printf("kernel: index %ld >= num_input_jobs %ld, returning\n", index, num_input_jobs);
// #endif
			return;
		}
		d_job += index;
		if (d_job->job_id < 100) {
			printf("invalid job passed to kernel with block,thread %d,%d index=%ld\n", blockIdx.x, threadIdx.x, index);
			d_job->d_print();
			return;
		}
		job::Job tmp_job = {};
		tmp_job.parent_job_id = d_job->job_id;
#ifdef TEST_WORKER_GPU
		if (index == 0) {
			printf("kernel: processing job %ld on index %ld\n", d_job->job_id, index);
			d_job->d_print();
			printf("tmp_job.parent_job_id = %ld\n", tmp_job.parent_job_id);
		}
#endif
		FrequencyMapIndex_t start = d_job->start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			printf("failure\n");
			return;
		}
		d_new_jobs += index * end;
		int32_t num_new_jobs = 0;
		for (FrequencyMapIndex_t i = start; i < end; i++) {
			tmp_job.is_sentence = false;
			tmp_job.finished = false;
			frequency_map::Result result = dict->d_compareFrequencyMaps_pip(
				&d_job->frequency_map,
				i,
				&tmp_job.frequency_map
			);
#ifdef TEST_WORKER_GPU
			if (index == 0) {
				printf("\n===\n");
				printf("compared frequency maps and got result: \n");
				printf(" "); d_job->frequency_map.d_print();
				printf("-"); dict->getFrequencyMapPointer(i)->d_print();
				printf("="); tmp_job.frequency_map.d_print();
				printf("\n===\n");
			}
#endif
			if (result == NO_MATCH) {
#ifdef TEST_WORKER_GPU
				if (index == 0) printf("kernel: job %ld: frequency map %d: no match, skipping\n", d_job->job_id, i);
#endif
				continue;
			}
			else if (result == COMPLETE_MATCH) {
#ifdef TEST_WORKER_GPU
				if (index == 0) printf("kernel: job %ld: frequency map %d: complete match\n", d_job->job_id, i);
#endif
				tmp_job.is_sentence = true;
			}
			else {
#ifdef TEST_WORKER_GPU
				if (index == 0) printf("kernel: job %ld: frequency map %d: incomplete match\n", d_job->job_id, i);
#endif
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
		int32_t worker_gpu_blocks = -1;
		int32_t worker_gpu_threads_per_block = -1;

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
			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: processing %ld jobs on device %d\n", p_count, device_id);
			fprintf(stderr, "Worker_GPU::doJob: max_input_jobs_per_iteration=%ld\n", max_input_jobs_per_iteration);
			#endif
			// launch kernel
			dim3 blocks(worker_gpu_blocks);
			dim3 threads(worker_gpu_threads_per_block);
			#ifdef TEST_WORKER_GPU
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
			gpuErrChk(cudaMemcpy(
				h_new_jobs_tmp,
				d_new_jobs,
				sizeof(Job) * max_new_jobs_per_job * p_count,
				cudaMemcpyDeviceToHost
			));
			int64_t num_total_new_jobs = 0;
			for (int64_t i = 0; i < max_input_jobs_per_iteration; i++) {
				int64_t num_new_jobs_i = h_num_new_jobs[i];
				if (num_new_jobs_i == 0) {
					continue;
				}
				#ifdef TEST_WORKER_GPU
				fprintf(stderr, "Worker_GPU::doJob: job %ld produced %ld new jobs\n", i, num_new_jobs_i);
				// read line to pause
				// std::string dummy;
				// std::getline(std::cin, dummy);
				#endif
				num_total_new_jobs += num_new_jobs_i;
				Job* tmp = h_new_jobs_tmp + (i * max_new_jobs_per_job);
				for (int64_t j = 0; j < num_new_jobs_i; j++) {
					last_result.new_jobs.push_back(*tmp);
					#if TEST_WORKER_GPU
					tmp->print();
					#endif
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
		int64_t estimatedMemoryUsage()
		{
			return sizeof(Dictionary) + max_input_jobs_per_iteration * (sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t));
			// sizeof(Dictionary) + max_input_jobs_per_iteration * (sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t)) < totalMem
			// max_input_jobs_per_iteration < (totalMem - sizeof(Dictionary))/(sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t))
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
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Allocating sizeof(Dictionary)=" << sizeof(Dictionary) << " bytes on device " << device_id << endl;
            gpuErrChk(cudaMalloc(&d_dict, sizeof(Dictionary)));
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Allocated d_dict on device " << device_id << endl;
            gpuErrChk(cudaMemcpy(
                d_dict,
                dict,
                sizeof(Dictionary),
                cudaMemcpyHostToDevice
            ));
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Copied Dictionary to device " << device_id << endl;

			cerr << "Allocating sizeof(Job)*" << max_input_jobs_per_iteration << "="
				 << sizeof(Job) * max_input_jobs_per_iteration
				 << " bytes for d_input_jobs on device " << device_id << endl;
			gpuErrChk(cudaMalloc(
				&d_input_jobs,
				sizeof(Job) * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Allocated d_input_jobs on device " << device_id << endl;

			cerr << "Allocating sizeof(Job)*" << (max_new_jobs_per_job * max_input_jobs_per_iteration)
				 << "="
				 << sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
				 << " bytes for d_new_jobs on device " << device_id << endl;
			gpuErrChk(cudaMalloc(
				&d_new_jobs,
				sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Allocated d_new_jobs on device " << device_id << endl;

			cerr << "Allocating sizeof(int64_t)*" << max_input_jobs_per_iteration << "="
				 << sizeof(int64_t) * max_input_jobs_per_iteration
				 << " bytes for d_num_new_jobs on device " << device_id << endl;
			gpuErrChk(cudaMalloc(
				&d_num_new_jobs,
				sizeof(int64_t) * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			cerr << "Allocated d_num_new_jobs on device " << device_id << endl;
			gpuErrChk(cudaMemset(
				d_num_new_jobs,
				0,
				sizeof(int64_t) * max_input_jobs_per_iteration
			));

			gpuErrChk(cudaDeviceSynchronize());

            while (true) {
                if (ready_to_start) {
					ready_to_start = false;
#ifdef TEST_WORKER_GPU
					fprintf(stderr, "Worker_GPU::loop: starting to process jobs on device %d\n", device_id);
#endif
                    doJobs();
#ifdef TEST_WORKER_GPU
					fprintf(stderr, "Worker_GPU::loop: done processing jobs, writing results to database\n");
					fprintf(stderr, "Worker_GPU::loop: number of new jobs: %ld\n", last_result.new_jobs.size());
#endif
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


		void setLimits()
		{
			// No allocations needed
			gpuErrChk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0));

			// Get our device info based on device id
			cudaDeviceProp deviceProp;
			gpuErrChk(cudaGetDeviceProperties(&deviceProp, device_id));

			// Based on our allocations and device info,
			//  set number of blocks and threads to utilize the GPU fully with the kernel's stack footprint in mind
			//  possibly adjusting the heap and stack sizes

			// Query kernel resource usage
			cudaFuncAttributes funcAttrib;
			gpuErrChk(cudaFuncGetAttributes(&funcAttrib, kernel));
			int64_t kernel_stack_size = funcAttrib.localSizeBytes;

			// print device total memory
			size_t freeMem, totalMem;
			gpuErrChk(cudaMemGetInfo(&freeMem, &totalMem));
			fprintf(stderr,
				"Device %d memory: free=%zu bytes, total=%zu bytes\n",
				device_id,
				freeMem,
				totalMem
			);

			max_input_jobs_per_iteration
				= (freeMem - sizeof(Dictionary))
					/
					(sizeof(Job)
						+ sizeof(Job) * max_new_jobs_per_job
						+ sizeof(int64_t)
						+ kernel_stack_size
					);
			cerr << "max_input_jobs_per_iteration set to " << max_input_jobs_per_iteration << " based on free memory of "
				 << freeMem << " bytes on device " << device_id << endl;

			//#endif
			worker_gpu_threads_per_block = deviceProp.maxThreadsPerBlock;
			// round max_input_jobs_per_iteration to the nearest multiple of worker_gpu_threads_per_block
			max_input_jobs_per_iteration
				= (max_input_jobs_per_iteration / worker_gpu_threads_per_block)
					* worker_gpu_threads_per_block;
			worker_gpu_blocks
				= max_input_jobs_per_iteration
					/ worker_gpu_threads_per_block;

			cerr << "Setting stack size per thread to kernel_stack_size = "
				 << kernel_stack_size << " bytes on device " << device_id << endl;
			gpuErrChk(cudaDeviceSetLimit(
				cudaLimitStackSize,
				kernel_stack_size
			));
			gpuErrChk(cudaDeviceSynchronize());

			// Query current limits
			size_t stackSize, heapSize;
			gpuErrChk(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			gpuErrChk(cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize));
			fprintf(stderr,
				"Device %d current limits: stack=%zu bytes, heap=%zu bytes\n",
				device_id,
				stackSize,
				heapSize
			);
		}

        Worker_GPU(
			Database* p_db, Dictionary* p_dict, int p_device_id
		) : Worker(p_db, p_dict)
        {
            device_id = p_device_id;
			gpuErrChk(cudaSetDevice(p_device_id));

			setLimits();

            h_thread = std::thread(&Worker_GPU::loop, this);
        }

        int64_t takeJobs(Job* buffer, int64_t max_length) override {
			int64_t to_take = std::min(
				(int64_t)numThreads(), max_length
			);
#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::takeJobs: taking %ld jobs on device %d\n", to_take, device_id);
#endif
            for (int64_t i = 0; i < to_take; i++) {
				h_unfinished_jobs.push_back(buffer++);
			}
            return to_take;
        }

        void doJobs_async() override {
			finished = false;
            ready_to_start = true;
        }

        int32_t numThreads() override {
#ifdef TEST_WORKER_GPU
			return 1;
#endif
            return worker_gpu_blocks * worker_gpu_threads_per_block;
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
#ifdef TEST_WORKER_GPU
			num_to_spawn = 1;
#endif
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
