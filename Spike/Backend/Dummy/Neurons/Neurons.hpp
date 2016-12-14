#pragma once

#include "Spike/Neurons/Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class Neurons : public virtual ::Backend::Neurons {
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
  } // namespace Dummy
} // namespace Backend

