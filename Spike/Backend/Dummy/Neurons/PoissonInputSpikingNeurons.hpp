#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class PoissonInputSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                                       public ::Backend::PoissonInputSpikingNeurons {
    public:
      virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
      // virtual void setup_random_states_on_device();
    };
  }
}
