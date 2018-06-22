#ifndef RateActivityMonitor_H
#define RateActivityMonitor_H


#include "../AcitivityMonitor/AcitivityMonitor.hpp"

class RateActivityMonitor; // forward definition

namespace Backend {
  class RateActivityMonitor : public virtual AcitivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(RateActivityMonitor);

    virtual void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds) = 0;
  };
}

class RateActivityMonitor : public AcitivityMonitor {
public:
  SPIKE_ADD_BACKEND_GETSET(RateActivityMonitor,
                           AcitivityMonitor);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  RateActivityMonitor(SpikingNeurons * neurons_parameter);
  ~RateActivityMonitor() override = default;

  void state_update(float current_time_in_seconds, float timestep);
  void initialise_count_neuron_spikes_recording_electrodes();
  void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::RateActivityMonitor> _backend;
};

#endif
