#pragma once

#include "Spike/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class CountNeuronSpikesRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodesCommon,
      public ::Backend::CountNeuronSpikesRecordingElectrodes {
    public:
      virtual void reset_state() {
        // TODO
      }
    };
  }
}
