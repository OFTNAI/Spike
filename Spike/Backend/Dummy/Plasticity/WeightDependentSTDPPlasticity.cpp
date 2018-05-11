#include "WeightDependentSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, WeightDependentSTDPPlasticity
);

namespace Backend {
  namespace Dummy {
    void WeightDependentSTDPPlasticity
  ::prepare() {
      STDPPlasticity::prepare();
    }

    void WeightDependentSTDPPlasticity
  ::reset_state() {
      STDPPlasticity::reset_state();
    }

    void WeightDependentSTDPPlasticity
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds, float timestep) {
    }
  }
}
