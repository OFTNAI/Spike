#pragma once

#include "Spike/Neurons/Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class Neurons : public virtual ::Backend::Neurons {
    public:
      ~Neurons() override = default;

      void prepare() override;
      void reset_state() override;

      void reset_current_injections() override;
    };
  } // namespace Dummy
} // namespace Backend

