#pragma once

#include "Spike/Neurons/Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class Neurons : public virtual ::Backend::Neurons {
    public:
      ~Neurons() override = default;

      void prepare() override;
      void reset_state() override;
    };
  } // namespace Dummy
} // namespace Backend

