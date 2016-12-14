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

      void prepare() override {
        RecordingElectrodes::prepare();
      }

      void reset_state() override {
        RecordingElectrodes::reset_state();
      }

      void push_data_front() override {
        RecordingElectrodes::push_data_front();
      }

      void pull_data_back() override {
        RecordingElectrodes::pull_data_back();
      }

      void copy_spikes_to_front() override {
      }

      void copy_spike_counts_to_front() override {
      }

      void collect_spikes_for_timestep
      (float current_time_in_seconds) override {
      }
    };
  }
}
