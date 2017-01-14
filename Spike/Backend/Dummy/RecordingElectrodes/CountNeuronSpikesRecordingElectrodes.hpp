#pragma once

#include "Spike/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class CountNeuronSpikesRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodes,
      public virtual ::Backend::CountNeuronSpikesRecordingElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(CountNeuronSpikesRecordingElectrodes);

      void prepare() override {
        RecordingElectrodes::prepare();
      }

      void reset_state() override {
        RecordingElectrodes::reset_state();
      }

      void add_spikes_to_per_neuron_spike_count
      (float current_time_in_seconds) override {
      }
    };
  }
}
