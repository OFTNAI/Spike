#pragma once

#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class ImagePoissonInputSpikingNeurons : public virtual ::Backend::Dummy::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);

      void update_membrane_potentials
      (float timestep, float current_time_in_seconds) override {
        // TODO
      }

      void check_for_neuron_spikes
      (float current_time_in_seconds, float timestep) override {
        // printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::check_for_neuron_spikes\n");
      }

      void reset_state() override {
        // printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::reset_state\n");
      }

      void prepare() override {
        printf("TODO Backend::Dummy::ImagePoissonInputSpikingNeurons::prepare\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
