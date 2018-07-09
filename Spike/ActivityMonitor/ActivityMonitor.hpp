#ifndef ActivityMonitor_H
#define ActivityMonitor_H

#include <string>
using namespace std;

class ActivityMonitor; // forward definition
#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Models/SpikingModel.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"


namespace Backend {
  class ActivityMonitor : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(ActivityMonitor);
  };
}

class ActivityMonitor : public virtual SpikeBase {
public:
  //ActivityMonitor(SpikingNeurons* neuron_set);
  ~ActivityMonitor() override = default;

  void init_backend(Context* ctx = _global_ctx) override;
  SPIKE_ADD_BACKEND_GETSET(ActivityMonitor, SpikeBase);

  SpikingNeurons* neurons = nullptr;
  SpikingModel* model = nullptr;
  virtual void state_update(float current_time_in_seconds, float timestep) = 0;
  virtual void final_update(float current_time_in_seconds, float timestep) = 0;
  virtual void reset_state() = 0;

private:
  std::shared_ptr<::Backend::ActivityMonitor> _backend;
};

#endif
