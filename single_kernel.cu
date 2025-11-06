/*
   class Job
   {

   public:
   JobID_t job_id;
   JobID_t parent_job_id;
   dictionary::frequency_map::FrequencyMap frequency_map;
// start represents both where to start the search, and what words (FrequencyMapIndex_t) this Job represents
dictionary::frequency_map::FrequencyMapIndex_t start;

}

 */

using namespace kernels;

__device__ inline d_processJobIndex(
	dictionary::Dictionary*& d_dictionary,
	Job& tmp,
	d_JobOutput*& d_output,
	int32_t& i
)
{
	dictionary::frequency_map::Result result;
	result = dictionary->d_compareFrequencyMaps_pip(
		&input.frequency_map,
		i,
		&tmp.frequency_map
	);
	switch (result) {
		case no_match:
			continue;
		case incomplete_match:
			tmp.start = i;
			d_output->addJob(tmp);
			break;
		case complete_match:
			tmp.start = i;
			d_output->addCompletedJob(tmp);
			break;
	}
}

__device__ d_processSingleJob(
	dictionary::Dictionary* d_dictionary,
	Job* d_input,
	d_JobOutput* d_output
)
{
	Job input = *d_input;
	Job tmp;
	tmp.parent_job_id = input.job_id;
	int32_t end = d_dictionary->getFrequencyMapSize();

	for (int32_t i = input.start; i < end; i++) {
		d_processJobIndex(
			d_dictionary,
			d_input,
			d_output,
			i
		);
	}

	*d_input = input;
}

__global__ d_processJobsWithThreads(
	dictionary::Dictionary* d_dictionary,
	Job* d_input,
	d_JobOutput* d_output
)
{
	d_processSingleJob(
		d_dictionary,
		d_input + threadIdx.x,

