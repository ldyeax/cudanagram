#include <memory>
#include "worker.hpp"
#include "job.hpp"
#include "frequency_map.hpp"

using namespace worker;

using job::Job;

class Worker_CPU : public Worker {
public:
	Worker_CPU(database::Database* p_db, dictionary::Dictionary* p_dict) : Worker(p_db, p_dict)
	{
		printf("Constructed CPU Worker\n");
	}
	Result doJob(job::Job input) override
	{
		frequency_map::FrequencyMap tmp = {};
		job::Job tmp_job = {};
		tmp_job.parent_job_id = input.job_id;
		FrequencyMapIndex_t start = input.start;
		FrequencyMapIndex_t end = dict->frequency_maps_length;
		if (start >= end) {
			throw;
		}

		Result ret;
		ret.num_new_jobs = 0;
		ret.num_found_sentences = 0;
		ret.new_jobs = new job::Job[end];
		ret.found_sentences = new job::Job[end];

		for (FrequencyMapIndex_t i = start; i < end; i++) {
			frequency_map::Result result = dict->h_compareFrequencyMaps_pip(
				&input.frequency_map,
				i,
				&tmp_job.frequency_map
			);
			if (result == INCOMPLETE_MATCH) {
				tmp_job.start = i;
				ret.new_jobs[ret.num_new_jobs++] = tmp_job;
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				ret.found_sentences[ret.num_found_sentences++] = tmp_job;
			}
		}

		return last_result = ret;
	}
};

Worker* worker::workerFactory_CPU(database::Database* db, dictionary::Dictionary* dict)
{
	return new Worker_CPU(db, dict);
}