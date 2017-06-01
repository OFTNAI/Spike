#pragma once

#include "Plasticity.hpp"
#include "Spike/Plasticity/Plasticity.hpp"
#include "Spike/Plasticity/WeightNormSpikingPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class WeightNormSpikingPlasticity : public virtual ::Backend::Dummy::Plasticity,
			   public virtual ::Backend::WeightNormSpikingPlasticity {
    public:
      void prepare() override;
      void weight_normalization() override;
      void reset_state() override;
    };
  }
}
