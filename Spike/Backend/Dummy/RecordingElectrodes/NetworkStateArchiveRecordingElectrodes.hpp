#pragma once

#include "Spike/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class NetworkStateArchiveRecordingElectrodes :
      public virtual ::Backend::Dummy::RecordingElectrodes,
      public virtual ::Backend::NetworkStateArchiveRecordingElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(NetworkStateArchiveRecordingElectrodes);

      void prepare() override;
      void reset_state() override;
    };
  }
}
