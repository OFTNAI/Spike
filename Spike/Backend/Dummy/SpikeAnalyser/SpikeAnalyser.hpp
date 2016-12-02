#pragma once

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"

namespace Backend {
  namespace Dummy {
    class SpikeAnalyserCommon : public virtual ::Backend::SpikeAnalyserCommon {
    public:
    };

    class SpikeAnalyser : public virtual ::Backend::Dummy::SpikeAnalyserCommon,
                          public ::Backend::SpikeAnalyser {
    public:
      MAKE_BACKEND_CONSTRUCTOR(SpikeAnalyser);

      virtual void store_spike_counts_for_stimulus_index(::SpikeAnalyser* front,
                                                         int stimulus_index) {
        printf("TODO store_spike_counts_for_stimulus_index\n");
      }

      virtual void push_data_front() {}
      virtual void pull_data_back() {}
    };
  }
}
