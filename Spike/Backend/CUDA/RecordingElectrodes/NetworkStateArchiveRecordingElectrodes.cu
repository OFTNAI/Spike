// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    void NetworkStateArchiveRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();
    }

    void NetworkStateArchiveRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();
    }

    void NetworkStateArchiveRecordingElectrodes::push_data_front() {
      RecordingElectrodes::push_data_front();
      frontend()->synapses->backend()->push_data_front();
    }

    void NetworkStateArchiveRecordingElectrodes::pull_data_back() {
      RecordingElectrodes::pull_data_back();
      frontend()->synapses->backend()->pull_data_back();
    }
  }
}
