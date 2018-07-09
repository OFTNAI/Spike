#ifndef RateActivityMonitor_H
#define RateActivityMonitor_H


#include "../ActivityMonitor/ActivityMonitor.hpp"

class RateActivityMonitor; // forward definition

namespace Backend {
  class RateActivityMonitor : public virtual ActivityMonitor {
  public:
    SPIKE_ADD_BACKEND_FACTORY(RateActivityMonitor);

    virtual void copy_spike_count_to_host() = 0;
    virtual void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds) = 0;
  };
}

class RateActivityMonitor : public ActivityMonitor {
public:
  SPIKE_ADD_BACKEND_GETSET(RateActivityMonitor,
                           ActivityMonitor);
  void init_backend(Context* ctx = _global_ctx) override;
  
  // Constructor/Destructor
  RateActivityMonitor(SpikingNeurons * neurons_parameter);
  ~RateActivityMonitor() override = default;
      
  int * per_neuron_spike_counts = nullptr;

  void prepare_backend_early() override;
  void state_update(float current_time_in_seconds, float timestep) override;
  void final_update(float current_time_in_seconds, float timestep) override;
  void reset_state() override;
  void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

private:
  std::shared_ptr<::Backend::RateActivityMonitor> _backend;
};

#endif
