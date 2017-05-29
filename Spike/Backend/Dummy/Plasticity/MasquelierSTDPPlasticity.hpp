#pragma once

#include "Spike/Plasticity/MasquelierSTDPPlasticity.hpp"
#include "STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDPPlasticity : public virtual ::Backend::Dummy::STDPPlasticity,
                           public virtual ::Backend::MasquelierSTDPPlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(MasquelierSTDPPlasticity);

      void prepare() override;
      void reset_state() override;

      void apply_stdp_to_synapse_weights(float current_time_in_seconds) override;
    };
  }
}
