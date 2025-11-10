using namespace worker;
using namespace worker_cpu;
using job::Job;

struct Result {
	Job* new_jobs;
	int32_t num_new_jobs;
	Job* found_sentences;
	int32_t num_found_sentences;
};

class Worker_CPU : public Worker {
private:
	database::Database* db;
	dictionary::Dictionary* dict;
public:
	void WriteResult(Result result)
	{
		if (result.num_new_jobs > 0) {
			db->writeJobs(
				result.new_jobs,
				result.num_new_jobs
			);
		}
		for (int32_t i = 0; i < result.num_found_sentences; i++) {
			db->writeCompleteSentence(
				result.found_sentences[i]
			);
		}
	}
	Worker_CPU(database::Database* p_db)
	{
		if (p_db == nullptr) {
			throw;
		}
		db = p_db;
	}
	Result doJob(job::Job input)
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
	}
}
