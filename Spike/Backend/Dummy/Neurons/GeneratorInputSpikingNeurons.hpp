#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    using GeneratorInputSpikingNeurons = ::Backend::Dummy::InputSpikingNeurons;
    // class GeneratorInputSpikingNeurons : public virtual ::Backend::Dummy::InputSpikingNeurons,
    //                                      public virtual ::Backend::GeneratorInputSpikingNeurons {
    // public:
    //   virtual void reset_state();
    // };
  }
}
