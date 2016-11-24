#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class LIFSpikingNeurons : public virtual ::Backend::Dummy::NeuronsCommon,
                              public ::Backend::LIFSpikingNeurons {
    public:
      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
        // printf("TODO Backend::Dummy::LIFSpikingNeurons::check_for_neuron_spikes\n");
      }

      virtual void reset_state() {
        // printf("TODO Backend::Dummy::LIFSpikingNeurons::reset_state\n");
      }

      virtual void prepare() {
        printf("TODO Backend::Dummy::LIFSpikingNeurons::prepare\n");
      }
    };
  }
}
