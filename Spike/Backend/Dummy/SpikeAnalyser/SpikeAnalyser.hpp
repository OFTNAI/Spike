#pragma once

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"

namespace Backend {
  namespace Dummy {
    class SpikeAnalyser : public virtual ::Backend::SpikeAnalyser {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(SpikeAnalyser);

      void prepare() override;
      void reset_state() override;

      void store_spike_counts_for_stimulus_index(int stimulus_index) override;
    };
  }
}
