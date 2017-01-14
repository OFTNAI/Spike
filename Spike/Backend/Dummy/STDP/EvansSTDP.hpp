#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDP : public virtual ::Backend::Dummy::STDP,
                      public virtual ::Backend::EvansSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EvansSTDP);

      void prepare() override;
      void reset_state() override;

      void update_synaptic_efficacies_or_weights
      (float current_time_in_seconds) override;
      void update_presynaptic_activities
      (float timestep, float current_time_in_seconds) override;
      void update_postsynaptic_activities
      (float timestep, float current_time_in_seconds) override;
    };
  }
}
