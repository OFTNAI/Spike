#pragma once

#include "Spike/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class NetworkStateArchiveRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodes,
      public virtual ::Backend::NetworkStateArchiveRecordingElectrodes {
    public:
      MAKE_BACKEND_CONSTRUCTOR(NetworkStateArchiveRecordingElectrodes);

      virtual void reset_state() {
        // TODO
      }

      virtual void copy_state_to_front(::NetworkStateArchiveRecordingElectrodes* front) {
        printf("TODO copy_state_to_front\n");
      }

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
