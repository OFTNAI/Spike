#pragma once

#include "Spike/STDP/STDP.hpp"

namespace Backend {
  namespace Dummy {
    class STDP : public virtual ::Backend::STDP {
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
