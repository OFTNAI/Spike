#pragma once

#include "STDPPlasticity.hpp"
#include "Spike/Plasticity/STDPPlasticity.hpp"
#include "Spike/Plasticity/WeightNormSTDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class WeightNormSTDPPlasticity : public virtual ::Backend::Dummy::Plasticity,
			   public virtual ::Backend::WeightNormSTDPPlasticity {
    public:
      void prepare() override;
      void weight_normalization() override;
      void reset_state() override;
    };
  }
}
