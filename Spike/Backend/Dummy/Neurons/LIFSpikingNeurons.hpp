#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class LIFSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                              public virtual ::Backend::LIFSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(LIFSpikingNeurons);

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

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::check_for_neuron_spikes;
      using ::Backend::Dummy::SpikingNeurons::update_membrane_potentials;
    };
  }
}
