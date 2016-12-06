// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    void NetworkStateArchiveRecordingElectrodes::prepare() {
      // set neurons_frontend and neurons_backend pointers:
      RecordingElectrodes::prepare();
    }

    void NetworkStateArchiveRecordingElectrodes::push_data_front() {
      frontend()->synapses->backend()->push_data_front();
    }
  }
}
