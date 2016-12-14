#pragma once

#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "PoissonInputSpikingNeurons.hpp"

namespace Backend {
  namespace Dummy {
    class ImagePoissonInputSpikingNeurons : public virtual ::Backend::Dummy::PoissonInputSpikingNeurons,
                                            public virtual ::Backend::ImagePoissonInputSpikingNeurons {
    public:
      MAKE_BACKEND_CONSTRUCTOR(ImagePoissonInputSpikingNeurons);

      void prepare() override {
        PoissonInputSpikingNeurons::prepare();
      }

      void reset_state() override {
        PoissonInputSpikingNeurons::reset_state();
      }

      void push_data_front() override {
        PoissonInputSpikingNeurons::push_data_front();
      }

      void pull_data_back() override {
        PoissonInputSpikingNeurons::pull_data_back();
      }

      void copy_rates_to_device() override {
      }
      
      // May want to override these when writing a new backend, or may not:
      using ::Backend::Dummy::SpikingNeurons::check_for_neuron_spikes;
      using ::Backend::Dummy::SpikingNeurons::update_membrane_potentials;
    };
  }
}
