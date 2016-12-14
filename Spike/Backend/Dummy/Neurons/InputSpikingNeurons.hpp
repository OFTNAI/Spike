#pragma once

#include "Spike/Neurons/InputSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class InputSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                                public virtual ::Backend::InputSpikingNeurons {
    public:
      void prepare() override {
        SpikingNeurons::prepare();
      }

      void reset_state() override {
        SpikingNeurons::reset_state();
      }

      void push_data_front() override {
        SpikingNeurons::push_data_front();
      }

      void pull_data_back() override {
        SpikingNeurons::pull_data_back();
      }
    };
  }
}
