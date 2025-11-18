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
	__global__ void printJobsKernel(
		Job* d_jobs,
		int64_t num_jobs
	)
	{
		printf("---printJobsKernel launched %p %ld\n", d_jobs, num_jobs);
		int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= num_jobs) {
			printf("printJobsKernel: index %ld >= num_jobs %ld, returning\n", index, num_jobs);
			return;
		}
		d_jobs += index;
		d_jobs->d_print();
	}
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
		//printf("---kernel launched\n");
		int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef TEST_WORKER_GPU
		if (index == 0) {
			printf("kernel launched with %d blocks of %d threads\n", gridDim.x, blockDim.x);
		}
#endif
		if (index >= num_input_jobs) {
 #ifdef TEST_WORKER_GPU
 			printf("kernel: index %ld >= num_input_jobs %ld, returning\n", index, num_input_jobs);
 #endif
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
		int32_t worker_gpu_threads_per_block = 1024;

		Dictionary* d_dict = nullptr;
		Job* d_input_jobs = nullptr;

		Job* unified_new_jobs = nullptr;
		int64_t* unified_num_new_jobs = nullptr;

		int64_t max_new_jobs_per_job;
		int64_t max_input_jobs_per_iteration;

		std::thread h_thread;
public:
        int device_id;

		virtual void writeNewJobsToDatabase() override
		{
		}

		void doJob(Job* d_input_jobs, int64_t p_count)
		{
			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: processing %ld jobs on device %d\n", p_count, device_id);
			fprintf(stderr, "Worker_GPU::doJob: max_input_jobs_per_iteration=%ld\n", max_input_jobs_per_iteration);
			fprintf(stderr, "Worker_GPU::doJob: d_input_jobs=%p\n", d_input_jobs);
			gpuErrChk(cudaDeviceSynchronize());
			printJobsKernel<<<1, 1>>>(
				d_input_jobs,
				p_count
			);
			gpuErrChk(cudaDeviceSynchronize());
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
				unified_new_jobs,
				unified_num_new_jobs,
				p_count
			);
			auto kernel_launch_error = cudaGetLastError();
			if (kernel_launch_error != cudaSuccess) {
				fprintf(
					stderr,
					"Worker_GPU::doJob: kernel launch failed on device %d: %s\n",
					device_id,
					cudaGetErrorString(kernel_launch_error)
				);
				throw std::runtime_error("Kernel launch failed");
			}
			gpuErrChk(cudaDeviceSynchronize());
			kernel_launch_error = cudaGetLastError();
			if (kernel_launch_error != cudaSuccess) {
				fprintf(
					stderr,
					"Worker_GPU::doJob: kernel launch failed on device %d: %s\n",
					device_id,
					cudaGetErrorString(kernel_launch_error)
				);
				throw std::runtime_error("Kernel launch failed");
			}
			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: kernel finished on device %d\n", device_id);
			#endif
			// copy number of new jobs back to host
			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: copying results numbers list back to host from device %d ..\n", device_id);
			#endif

			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: copied results numbers list back to host from device %d\n", device_id);
			#endif
			//Job* h_write_pointer = new_jobs_buffer;
			for (int64_t i = 0; i < max_input_jobs_per_iteration; i++) {
				int64_t num_new_jobs_i = unified_num_new_jobs[i];
				if (num_new_jobs_i == 0) {
					continue;
				}
				#ifdef TEST_WORKER_GPU
				fprintf(stderr, "Worker_GPU::doJob: job %ld produced %ld new jobs\n", i, num_new_jobs_i);
				// read line to pause
				// std::string dummy;
				// std::getline(std::cin, dummy);
				#endif
				num_new_jobs += num_new_jobs_i;
				//Job* tmp = h_new_jobs_tmp + (i * max_new_jobs_per_job);
				// gpuErrChk(cudaMemcpy(
				// 	h_write_pointer,
				// 	d_new_jobs + (i * max_new_jobs_per_job),
				// 	sizeof(Job) * num_new_jobs_i,
				// 	cudaMemcpyDeviceToHost
				// ));
				// This is unified memory now
				database->writeNewJobs(
					unified_new_jobs + (i * max_new_jobs_per_job),
					num_new_jobs_i
				);
				//h_write_pointer += num_new_jobs_i;
			}
			#ifdef TEST_WORKER_GPU
			fprintf(stderr, "Worker_GPU::doJob: total new jobs produced: %ld\n", num_new_jobs);
			// read line to pause
			#endif
		}

		void doJobs()
        {
			int64_t jobs_done = 0;
			num_new_jobs = 0;
			while (jobs_done < num_unfinished_jobs) {
				memset(unified_num_new_jobs, 0, sizeof(int64_t) * max_input_jobs_per_iteration);

				int64_t jobs_start = jobs_done;
				int64_t jobs_end = jobs_start + max_input_jobs_per_iteration;
				if (jobs_end > num_unfinished_jobs) {
					jobs_end = num_unfinished_jobs;
				}
				int64_t num_input_jobs = jobs_end - jobs_start;
				#ifdef TEST_WORKER_GPU
				cerr << "Copying input jobs to device " << device_id << ": jobs " << jobs_start << " to " << jobs_end << " ("
					 << num_input_jobs << " jobs).." << endl;
				#endif
				// copy input jobs to device
				gpuErrChk(cudaMemcpy(
					d_input_jobs,
					&unfinished_jobs[jobs_start],
					sizeof(Job) * num_input_jobs,
					cudaMemcpyHostToDevice
				));
				#ifdef TEST_WORKER_GPU
				cerr << "Copied input jobs to device " << device_id << endl;
				#endif
				// process each job
				doJob(
					d_input_jobs,
					num_input_jobs
				);
				#ifdef TEST_WORKER_GPU
				cerr << "finished doJob on device " << device_id << endl;
				#endif
				jobs_done += num_input_jobs;
			}
			//cerr << "End of doJobs num_new_jobs=" << num_new_jobs << endl;
        }
		int64_t estimatedMemoryUsage()
		{
			return sizeof(Dictionary) + max_input_jobs_per_iteration * (sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t));
			// sizeof(Dictionary) + max_input_jobs_per_iteration * (sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t)) < totalMem
			// max_input_jobs_per_iteration < (totalMem - sizeof(Dictionary))/(sizeof(Job) + sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t))
		}


		void setLimits()
		{
			// Initialize max_new_jobs_per_job based on dictionary size
			max_new_jobs_per_job = dictionary->frequency_maps_length;

			// No allocations needed
			// cerr << "Setting malloc heap size to 0 on device " << device_id << endl;
			// gpuErrChk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0));
			gpuErrChk(cudaDeviceSynchronize());

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

			// max_input_jobs_per_iteration
			// 	= (freeMem - sizeof(Dictionary))
			// 		/
			// 		(
			// 			sizeof(Job)
			// 			// + sizeof(Job) * max_new_jobs_per_job - unified
			// 			// + sizeof(int64_t) - unified
			// 		);
			// Now determining max_input_jobs_per_iteration beforehand in init()
			//  based on how many jobs could fit into half the memory
			//max_input_jobs_per_iteration = max_input_jobs_per_iteration * 3L / 4L;
			// cerr << "max_input_jobs_per_iteration set to " << max_input_jobs_per_iteration << " based on free memory of "
			// 	 << freeMem << " bytes on device " << device_id << endl;

			//#endif
			//worker_gpu_threads_per_block = deviceProp.maxThreadsPerBlock;

			// Round to multiple of threads per block
			// max_input_jobs_per_iteration
			// 	= (max_input_jobs_per_iteration / worker_gpu_threads_per_block)
			// 		* worker_gpu_threads_per_block;
			worker_gpu_blocks
				= max_input_jobs_per_iteration
					/ worker_gpu_threads_per_block;

			// Note: kernel_stack_size (funcAttrib.localSizeBytes) is static local memory,
			// not dynamic stack. Don't artificially limit cudaLimitStackSize based on it.
			// CUDA manages stack separately from device memory allocations.
			cerr << "Kernel uses " << kernel_stack_size << " bytes of static local memory per thread on device " << device_id << endl;

			/*cudaDeviceSetLimit(
				cudaLimitStackSize,
				4096
			);*/

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

		void init() override
		{
			gpuErrChk(cudaSetDevice(device_id));
			gpuErrChk(cudaDeviceSynchronize());

			#ifdef TEST_WORKER_GPU
			cerr << "Allocating sizeof(Dictionary)=" << sizeof(Dictionary) << " bytes on device " << device_id << endl;
			#endif
            gpuErrChk(cudaMalloc(&d_dict, sizeof(Dictionary)));
			gpuErrChk(cudaDeviceSynchronize());

			#ifdef TEST_WORKER_GPU
			cerr << "Allocated d_dict on device " << device_id << endl;
			#endif
            gpuErrChk(cudaMemcpy(
                d_dict,
                dictionary,
                sizeof(Dictionary),
                cudaMemcpyHostToDevice
            ));
			gpuErrChk(cudaDeviceSynchronize());
			#ifdef TEST_WORKER_GPU
			cerr << "Copied Dictionary to device " << device_id << endl;
			#endif

			// get free cuda memory
			size_t freeMemBefore, totalMem;
			gpuErrChk(cudaMemGetInfo(&freeMemBefore, &totalMem));
			fprintf(stderr,
				"Device %d memory before allocations: free=%zu bytes, total=%zu bytes\n",
				device_id,
				freeMemBefore,
				totalMem
			);
			int64_t mem_for_calculation = freeMemBefore * 8L / 9;
			int64_t num_jobs_in_whole_memory = mem_for_calculation / sizeof(Job);

			max_new_jobs_per_job = dictionary->frequency_maps_length;

			max_input_jobs_per_iteration = num_jobs_in_whole_memory / (1 + max_new_jobs_per_job);

			int64_t expected_unified_usage =
				sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
				+ sizeof(int64_t) * max_input_jobs_per_iteration;
			// 32GB cap
			if (expected_unified_usage > 32 * 1024 * 1024 * 1024) {
				max_input_jobs_per_iteration =
					(32L * 1024L * 1024L * 1024L)
					/
					(sizeof(Job) * max_new_jobs_per_job + sizeof(int64_t));
			}

			cerr << "max_input_jobs_per_iteration set to " << max_input_jobs_per_iteration
				 << " on device " << device_id << endl;
			expected_unified_usage =
				sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
				+ sizeof(int64_t) * max_input_jobs_per_iteration;
			cerr << "Expected unified memory usage: " << expected_unified_usage << " bytes on device " << device_id << endl;

			// max input jobs + max input jobs * max new jobs per job = total jobs that would fit
			// max input jobs (1 + max new jobs per job) = total jobs
			// total jobs / (1 + max new jobs per job) = max input jobs



			//#ifdef TEST_WORKER_GPU
			cerr << "Allocating sizeof(Job)*" << max_input_jobs_per_iteration << "="
				 << sizeof(Job) * max_input_jobs_per_iteration
				 << " bytes for d_input_jobs on device " << device_id << endl;
			//#endif
			gpuErrChk(cudaMalloc(
				&d_input_jobs,
				sizeof(Job) * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			#ifdef TEST_WORKER_GPU
			cerr << "Allocated d_input_jobs on device " << device_id << endl;
			#endif

			setLimits();

			//#ifdef TEST_WORKER_GPU
			cerr << "Allocating sizeof(Job)*" << (max_new_jobs_per_job * max_input_jobs_per_iteration)
				 << "="
				 << sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
				 << " bytes for unified_new_jobs on device " << device_id << endl;
			//#endif
			gpuErrChk(cudaMallocManaged(
				&unified_new_jobs,
				sizeof(Job) * max_new_jobs_per_job * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			#ifdef TEST_WORKER_GPU
			cerr << "Allocated d_new_jobs on device " << device_id << endl;
			#endif
			//#ifdef TEST_WORKER_GPU
			cerr << "Allocating sizeof(int64_t)*" << max_input_jobs_per_iteration << "="
				 << sizeof(int64_t) * max_input_jobs_per_iteration
				 << " bytes for unified_num_new_jobs on device " << device_id << endl;
			//#endif
			gpuErrChk(cudaMallocManaged(
				&unified_num_new_jobs,
				sizeof(int64_t) * max_input_jobs_per_iteration
			));
			gpuErrChk(cudaDeviceSynchronize());
			#ifdef TEST_WORKER_GPU
			cerr << "Allocated unified_num_new_jobs " << device_id << endl;
			#endif
			cerr << "Allocating sizeof(int64_t)*" << max_input_jobs_per_iteration << "="
				 << sizeof(int64_t) * max_input_jobs_per_iteration
				 << " bytes for unified_num_new_jobs on device " << device_id << endl;
			gpuErrChk(cudaMemset(
				unified_num_new_jobs,
				0,
				sizeof(int64_t) * max_input_jobs_per_iteration
			));

			gpuErrChk(cudaDeviceSynchronize());
		}


        Worker_GPU(
			int p_device_id,
			Dictionary* p_dict,
			Job* p_initial_jobs,
			int64_t p_num_initial_jobs,
			shared_ptr<vector<Job>> non_sentence_finished_jobs
		)
		: Worker(
			p_dict,
			p_initial_jobs,
			p_num_initial_jobs,
			non_sentence_finished_jobs
		)
        {
            device_id = p_device_id;
			cerr << "Constructed Worker_GPU on device " << device_id << endl;
        }

//         int64_t numThreads() override {
// #ifdef TEST_WORKER_GPU
// 			return 1;
// #endif
//             return worker_gpu_blocks * worker_gpu_threads_per_block;
//         }


    };
    class WorkerFactory_GPU : public WorkerFactory {
    public:
		/**
		 * CPU worker factory would return number of system threads,
		 * GPU would return number of CUDA threads it can use across
		 *  all available GPUs
		 **/
		virtual int64_t getTotalThreads() override
		{
			// int32_t num_devices = deviceCount();
			// int64_t total_threads = 0;
			// for (int32_t i = 0; i < num_devices; i++) {
			// 	cudaDeviceProp deviceProp;
			// 	gpuErrChk(cudaGetDeviceProperties(&deviceProp, i));
			// 	total_threads += deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount;
			// }
			// return total_threads;
			return deviceCount() * 768L * 1024L;
		}
        virtual int64_t spawn(
			atomic<Worker*>* buffer,
			Dictionary* dict,
			Job* initial_jobs,
			int64_t num_initial_jobs,
			shared_ptr<vector<Job>> non_sentence_finished_jobs
        ) override {
            int num_devices = deviceCount();
			cerr << "GPU Spawn: " << num_devices << " devices detected" << endl;
			Job* device_initial_jobs = initial_jobs;
			int64_t jobs_per_device = num_initial_jobs / num_devices;
			cerr << "GPU Spawn : " << jobs_per_device << " jobs per device" << endl;
			if (jobs_per_device <= 0) {
				cerr << "Jobs per device = " << jobs_per_device << endl;
				exit(1);
			}
			for (int i = 0; i < num_devices; i++) {
				if (jobs_per_device <= 0) {
					cerr << "No jobs to assign to device " << i << ", breaking spawn loop" << endl;
					break;
				}
				std::thread t2([=]{
					buffer[i].store(new Worker_GPU(
						i,
						dict,
						device_initial_jobs + jobs_per_device * i,
						jobs_per_device,
						non_sentence_finished_jobs
					));
					fprintf(stderr, "Started Worker_GPU on device %d at %p\n", i, buffer[i].load());
					buffer[i].load()->start();
				});
				t2.detach();
			}
			return num_devices;
        }
    };
}


WorkerFactory* worker::getWorkerFactory_GPU()
{
	static WorkerFactory* ret = new worker_GPU::WorkerFactory_GPU();
	return ret;
}
