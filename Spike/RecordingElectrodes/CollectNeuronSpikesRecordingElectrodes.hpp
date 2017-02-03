#ifndef CollectNeuronSpikesRecordingElectrodes_H
#define CollectNeuronSpikesRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.hpp"

class CollectNeuronSpikesRecordingElectrodes; // forward definition

namespace Backend {
  class CollectNeuronSpikesRecordingElectrodes : public virtual RecordingElectrodes {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CollectNeuronSpikesRecordingElectrodes);

    virtual void copy_spikes_to_front() = 0;
    virtual void copy_spike_counts_to_front() = 0;
    virtual void collect_spikes_for_timestep(float current_time_in_seconds) = 0;
  };
}

struct Collect_Neuron_Spikes_Optional_Parameters {

	Collect_Neuron_Spikes_Optional_Parameters(): number_of_timesteps_per_device_spike_copy_check(50), device_spike_store_size_multiple_of_total_neurons(52), proportion_of_device_spike_store_full_before_copy(0.2), human_readable_storage(false) {}

	int number_of_timesteps_per_device_spike_copy_check;
	int device_spike_store_size_multiple_of_total_neurons;
	float proportion_of_device_spike_store_full_before_copy;
	bool human_readable_storage;

};




class CollectNeuronSpikesRecordingElectrodes : public RecordingElectrodes {
public:
  SPIKE_ADD_BACKEND_GETSET(CollectNeuronSpikesRecordingElectrodes,
                           RecordingElectrodes);
  void init_backend(Context* ctx = _global_ctx) override;

  // Variables
  int size_of_device_spike_store;
  int total_number_of_spikes_stored_on_host;

  // Host Pointers
  Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters = nullptr;
  int* neuron_ids_of_stored_spikes_on_host = nullptr;
  int* total_number_of_spikes_stored_on_device = nullptr;
  float* time_in_seconds_of_stored_spikes_on_host = nullptr;
  int* reset_neuron_ids = nullptr;
  float* reset_neuron_times = nullptr;

  // Constructor/Destructor
  CollectNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
  ~CollectNeuronSpikesRecordingElectrodes() override;

  void initialise_collect_neuron_spikes_recording_electrodes(Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters_param);
  void allocate_pointers_for_spike_store();

  void collect_spikes_for_timestep(float current_time_in_seconds);
  void copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch);
  void write_spikes_to_file(int epoch_number, bool isTrained);

  void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

  void delete_and_reset_collected_spikes();

private:
  std::shared_ptr<::Backend::CollectNeuronSpikesRecordingElectrodes> _backend;

};

#endif
