using namespace worker_cpu;

class Worker_CPU : public Worker {
private:
	database::Database* db;
	dictionary::Dictionary* dict;
public:
	Worker_CPU(database::Database* p_db)
	{
		if (p_db == nullptr) {
			throw;
		}
		db = p_db;
	}
	void doJob(job::Job input)
	{
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
				// todo: batching?
				db->writeJob(tmp_job);
			}
			else if (result == COMPLETE_MATCH) {
				tmp_job.start = i;
				db->writeCompleteSentence(tmp_job);
			}
		}
	}
}
