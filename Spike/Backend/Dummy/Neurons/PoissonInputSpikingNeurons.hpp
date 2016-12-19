#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class PoissonInputSpikingNeurons : public virtual ::Backend::Dummy::InputSpikingNeurons,
                                       public virtual ::Backend::PoissonInputSpikingNeurons {
    public:
      PoissonInputSpikingNeurons() = default;
      MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);

      void prepare() override {
        InputSpikingNeurons::prepare();
      }

      void reset_state() override {
        InputSpikingNeurons::reset_state();
      }

      void push_data_front() override {
        InputSpikingNeurons::push_data_front();
      }

      void pull_data_back() override {
        InputSpikingNeurons::pull_data_back();
      }
    };
  }
}
