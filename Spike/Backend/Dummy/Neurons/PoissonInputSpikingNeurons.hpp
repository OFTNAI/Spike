#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    using PoissonInputSpikingNeurons = ::Backend::Dummy::InputSpikingNeurons;
    // class PoissonInputSpikingNeurons : public virtual ::Backend::Dummy::InputSpikingNeurons,
    //                                    public virtual ::Backend::PoissonInputSpikingNeurons {
    // public:
    //   virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
    //   // virtual void setup_random_states_on_device();
    // };
  }
}
