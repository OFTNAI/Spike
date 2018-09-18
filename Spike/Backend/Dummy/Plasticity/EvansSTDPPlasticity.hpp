#pragma once

#include "Spike/Plasticity/EvansSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                      public virtual ::Backend::EvansSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EvansSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void update_synaptic_efficacies_or_weights
      (float current_time_in_seconds, float timestep) override;
    };
  }
}
