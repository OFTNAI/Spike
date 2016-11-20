#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class ImagePoissonInputSpikingNeurons : public ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
        printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::check_for_neuron_spikes\n");
      }

      virtual void reset_state() {
        printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::reset_state\n");
      }

      virtual void prepare() {
        printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::prepare\n");
      }
    };
  }
}
