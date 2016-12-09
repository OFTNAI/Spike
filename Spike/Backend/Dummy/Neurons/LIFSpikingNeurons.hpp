#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class LIFSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                              public virtual ::Backend::LIFSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(LIFSpikingNeurons);

      void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override {
        // printf("TODO Backend::Dummy::LIFSpikingNeurons::check_for_neuron_spikes\n");
      }

      void update_membrane_potentials(float timestep, float current_time_in_seconds) override {
        // TODO
      }

      void reset_state() override {
        // printf("TODO Backend::Dummy::LIFSpikingNeurons::reset_state\n");
      }

      void prepare() override {
        printf("TODO Backend::Dummy::LIFSpikingNeurons::prepare\n");
      }

      void push_data_front() override {}
      void pull_data_back() override {}
    };
  }
}
