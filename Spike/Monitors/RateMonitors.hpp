#ifndef RateMonitors_H
#define RateMonitors_H


#include "../Monitors/Monitors.hpp"

class RateMonitors; // forward definition

namespace Backend {
  class RateMonitors : public virtual Monitors {
  public:
    SPIKE_ADD_BACKEND_FACTORY(RateMonitors);

    virtual void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds) = 0;
  };
}

class RateMonitors : public Monitors {
public:
  SPIKE_ADD_BACKEND_GETSET(RateMonitors,
                           Monitors);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  RateMonitors(SpikingNeurons * neurons_parameter);
  ~RateMonitors() override = default;

  void state_update(float current_time_in_seconds, float timestep);
  void initialise_count_neuron_spikes_recording_electrodes();
  void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::RateMonitors> _backend;
};

#endif
