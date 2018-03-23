#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class RecordingElectrodes : public virtual ::Backend::RecordingElectrodes {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
