#pragma once

#include "Plasticity.hpp"
#include "Spike/Plasticity/Plasticity.hpp"
#include "Spike/Plasticity/STDPPlasticity.hpp"

namespace Backend {
  namespace Dummy {
    class STDPPlasticity : public virtual ::Backend::Dummy::Plasticity,
			   public virtual ::Backend::STDPPlasticity {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
