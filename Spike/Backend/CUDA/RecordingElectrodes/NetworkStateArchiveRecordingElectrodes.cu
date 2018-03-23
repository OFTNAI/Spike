// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, NetworkStateArchiveRecordingElectrodes);

namespace Backend {
  namespace CUDA {
    void NetworkStateArchiveRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();
    }

    void NetworkStateArchiveRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();
    }
  }
}
