#pragma once

#include "Spike/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class CollectNeuronSpikesRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodes,
      public virtual ::Backend::CollectNeuronSpikesRecordingElectrodes {
    public:
      MAKE_BACKEND_CONSTRUCTOR(CollectNeuronSpikesRecordingElectrodes);

      virtual void reset_state() {
        // TODO
      }
      virtual void copy_spikes_to_front(::CollectNeuronSpikesRecordingElectrodes* front) {
        printf("TODO copy_spikes_to_front\n");
      }

      virtual void copy_spike_counts_to_front(::CollectNeuronSpikesRecordingElectrodes* front) {
        printf("TODO copy_spike_counts_to_front\n");
      }

      virtual void collect_spikes_for_timestep(::CollectNeuronSpikesRecordingElectrodes* front,
                                               float current_time_in_seconds) {
        printf("TODO collect_spikes_for_timestep\n");
      }

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
