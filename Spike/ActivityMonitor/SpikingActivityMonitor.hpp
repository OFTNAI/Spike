#ifndef SpikingActivityMonitor_H
#define SpikingActivityMonitor_H


#include "../ActivityMonitor/ActivityMonitor.hpp"

class SpikingActivityMonitor; // forward definition

struct spike_monitor_advanced_parameters {

  spike_monitor_advanced_parameters(): number_of_timesteps_per_device_spike_copy_check(50), device_spike_store_size_multiple_of_total_neurons(60), proportion_of_device_spike_store_full_before_copy(0.2) {}

  int number_of_timesteps_per_device_spike_copy_check;
  int device_spike_store_size_multiple_of_total_neurons;
  float proportion_of_device_spike_store_full_before_copy;

};




namespace Backend {
  class SpikingActivityMonitor : public virtual ActivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(SpikingActivityMonitor);

    virtual void copy_spikes_to_front() = 0;
    virtual void copy_spikecount_to_front() = 0;
    virtual void collect_spikes_for_timestep(float current_time_in_seconds, float timestep) = 0;
  };
}


class SpikingActivityMonitor : public ActivityMonitor {
public:
  void init_backend(Context* ctx = _global_ctx) override;
  SPIKE_ADD_BACKEND_GETSET(SpikingActivityMonitor,
                           ActivityMonitor);

  // Variables
  int size_of_device_spike_store;
  int total_number_of_spikes_stored_on_host;

  // Host Pointers
  SpikingNeurons * neurons;
  spike_monitor_advanced_parameters * advanced_parameters = nullptr;
  int* neuron_ids_of_stored_spikes_on_host = nullptr;
  float* spike_times_of_stored_spikes_on_host = nullptr;
  int* total_number_of_spikes_stored_on_device = nullptr;
  int* reset_neuron_ids = nullptr;
  float* reset_neuron_times = nullptr;

  // Constructor/Destructor
  SpikingActivityMonitor(SpikingNeurons * neurons_parameter);
  ~SpikingActivityMonitor() override;
  
  void prepare_backend_early() override;

  void initialise_collect_neuron_spikes_recording_electrodes();
  void allocate_pointers_for_spike_store();

  void state_update(float current_time_in_seconds, float timestep) override;
  void final_update(float current_time_in_seconds, float timestep) override;
  void reset_state() override;

  void copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, float timestep, bool force=false);

  void save_spikes_as_txt(string path, string prefix="");
  void save_spikes_as_binary(string path, string prefix="");


private:
  std::shared_ptr<::Backend::SpikingActivityMonitor> _backend;

};

#endif
