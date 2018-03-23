#include "SpikeAnalyser.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, SpikeAnalyser);

namespace Backend {
  namespace Dummy {
    void SpikeAnalyser::prepare() {
    }

    void SpikeAnalyser::reset_state() {
    }

    void SpikeAnalyser::store_spike_counts_for_stimulus_index
    (int stimulus_index) {
    }
  }
}
