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

      void prepare() override {
        RecordingElectrodes::prepare();
      }

      void reset_state() override {
        RecordingElectrodes::reset_state();
      }

      void push_data_front() override {
        RecordingElectrodes::push_data_front();
      }

      void pull_data_back() override {
        RecordingElectrodes::pull_data_back();
      }
    };
  }
}
