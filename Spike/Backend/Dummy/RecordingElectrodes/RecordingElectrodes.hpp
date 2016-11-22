#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class RecordingElectrodesCommon : public virtual ::Backend::RecordingElectrodesCommon {
    public:
    };

    class RecordingElectrodes : public virtual ::Backend::Dummy::RecordingElectrodesCommon,
                                public ::Backend::RecordingElectrodes {
    public:
    };
  }
}
