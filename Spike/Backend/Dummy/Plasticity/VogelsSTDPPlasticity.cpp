#include "VogelsSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, VogelsSTDPPlasticity
);

namespace Backend {
  namespace Dummy {
    void VogelsSTDPPlasticity
  ::prepare() {
      STDPPlasticity::prepare();
    }

    void VogelsSTDPPlasticity
  ::reset_state() {
      STDPPlasticity::reset_state();
    }

    void VogelsSTDPPlasticity
  ::apply_stdp_to_synapse_weights
    (float current_time_in_seconds, float timestep) {
    }
  }
}
