#pragma once

#include "Spike/RecordingElectrodes/RecordingElectrodes.hpp"

namespace Backend {
  namespace Dummy {
    class RecordingElectrodes : public virtual ::Backend::RecordingElectrodes {
    public:
      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };
  }
}
