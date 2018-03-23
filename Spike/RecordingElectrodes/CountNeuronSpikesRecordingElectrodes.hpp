#ifndef CountNeuronSpikesRecordingElectrodes_H
#define CountNeuronSpikesRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.hpp"

class CountNeuronSpikesRecordingElectrodes; // forward definition

namespace Backend {
  class CountNeuronSpikesRecordingElectrodes : public virtual RecordingElectrodes {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CountNeuronSpikesRecordingElectrodes);

    virtual void add_spikes_to_per_neuron_spike_count
    (float current_time_in_seconds) = 0;
  };
}

class CountNeuronSpikesRecordingElectrodes : public RecordingElectrodes {
public:
  SPIKE_ADD_BACKEND_GETSET(CountNeuronSpikesRecordingElectrodes,
                           RecordingElectrodes);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  CountNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter,
                                       SpikingSynapses * synapses_parameter,
                                       string full_directory_name_for_simulation_data_files_param,
                                       const char * prefix_string_param);
  ~CountNeuronSpikesRecordingElectrodes() override = default;

  void initialise_count_neuron_spikes_recording_electrodes();

  void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::CountNeuronSpikesRecordingElectrodes> _backend;
};

#endif
