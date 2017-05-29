#pragma once

#include "Spike/Plasticity/Plasticity.hpp"

namespace Backend {
  namespace Dummy {
    class Plasticity : public virtual ::Backend::Plasticity {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
