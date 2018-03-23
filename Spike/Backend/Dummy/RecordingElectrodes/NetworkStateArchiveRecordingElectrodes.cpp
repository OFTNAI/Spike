#include "NetworkStateArchiveRecordingElectrodes.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, NetworkStateArchiveRecordingElectrodes);

namespace Backend {
  namespace Dummy {
    void NetworkStateArchiveRecordingElectrodes::prepare() {
      RecordingElectrodes::prepare();
    }

    void NetworkStateArchiveRecordingElectrodes::reset_state() {
      RecordingElectrodes::reset_state();
    }
  }
}
