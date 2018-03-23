#include "CountNeuronSpikesRecordingElectrodes.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CountNeuronSpikesRecordingElectrodes);

namespace Backend {
  namespace Dummy {
    void CountNeuronSpikesRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();
    }

    void CountNeuronSpikesRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();
    }

    void CountNeuronSpikesRecordingElectrodes::add_spikes_to_per_neuron_spike_count
    (float current_time_in_seconds) {
    }
  }
}
