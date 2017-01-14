#include "CollectNeuronSpikesRecordingElectrodes.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, CollectNeuronSpikesRecordingElectrodes);

namespace Backend {
  namespace Dummy {
    void CollectNeuronSpikesRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();
    }

    void CollectNeuronSpikesRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();
    }

    void CollectNeuronSpikesRecordingElectrodes::copy_spikes_to_front() {
    }

    void CollectNeuronSpikesRecordingElectrodes::copy_spike_counts_to_front() {
    }

    void CollectNeuronSpikesRecordingElectrodes::collect_spikes_for_timestep
    (float current_time_in_seconds) {
    }
  }
}
