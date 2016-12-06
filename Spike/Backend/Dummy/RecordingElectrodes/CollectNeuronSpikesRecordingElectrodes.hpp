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

      void reset_state() override {
        // TODO
      }

      void copy_spikes_to_front() override {
        printf("TODO copy_spikes_to_front\n");
      }

      void copy_spike_counts_to_front() override {
        printf("TODO copy_spike_counts_to_front\n");
      }

      void collect_spikes_for_timestep
      (float current_time_in_seconds) override{
        printf("TODO collect_spikes_for_timestep\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
