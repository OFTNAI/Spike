#pragma once

#include "Spike/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class NetworkStateArchiveRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodesCommon,
      public ::Backend::NetworkStateArchiveRecordingElectrodes {
    public:
      virtual void reset_state() {
        // TODO
      }

      virtual void copy_state_to_front(::NetworkStateArchiveRecordingElectrodes* front) {
        printf("TODO copy_state_to_front\n");
      }
    };
  }
}
