#pragma once

#include "Spike/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class CollectNeuronSpikesRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodes,
      public virtual ::Backend::CollectNeuronSpikesRecordingElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CollectNeuronSpikesRecordingElectrodes);

      void prepare() override;
      void reset_state() override;

      void copy_spikes_to_front() override;
      void copy_spike_counts_to_front() override;
      void collect_spikes_for_timestep
      (float current_time_in_seconds) override;
    };
  }
}
