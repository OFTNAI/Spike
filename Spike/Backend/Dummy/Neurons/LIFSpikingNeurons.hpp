#pragma once

#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class LIFSpikingNeurons : public virtual ::Backend::Dummy::SpikingNeurons,
                              public virtual ::Backend::LIFSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(LIFSpikingNeurons);

      void prepare() override {
        SpikingNeurons::prepare();
      }

      void reset_state() override {
        SpikingNeurons::reset_state();
      }

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::check_for_neuron_spikes;
      using ::Backend::Dummy::SpikingNeurons::update_membrane_potentials;
    };
  }
}
