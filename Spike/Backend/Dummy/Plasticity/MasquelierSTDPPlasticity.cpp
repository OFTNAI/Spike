#include "MasquelierSTDPPlasticity.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, MasquelierSTDPPlasticity);

namespace Backend {
  namespace Dummy {
    void MasquelierSTDPPlasticity::prepare() {
      STDPPlasticity::prepare();
    }

    void MasquelierSTDPPlasticity::reset_state() {
      STDPPlasticity::reset_state();
    }

    void MasquelierSTDPPlasticity::apply_stdp_to_synapse_weights
    (float current_time_in_seconds) {
    }
  }
}
