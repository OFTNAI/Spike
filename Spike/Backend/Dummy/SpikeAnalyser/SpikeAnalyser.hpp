#pragma once

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"

namespace Backend {
  namespace Dummy {
    class SpikeAnalyser : public virtual ::Backend::SpikeAnalyser {
    public:
      MAKE_BACKEND_CONSTRUCTOR(SpikeAnalyser);

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }

      void store_spike_counts_for_stimulus_index(int stimulus_index) override {
      }
    };
  }
}
