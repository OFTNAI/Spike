#pragma once

#include "Spike/Neurons/Neurons.hpp"

namespace Backend {
  namespace Dummy {
    class NeuronsCommon : public virtual ::Backend::NeuronsCommon {
    public:
    };

    class Neurons : public virtual ::Backend::Dummy::NeuronsCommon,
                    public ::Backend::Neurons {
    public:
      virtual void prepare() {}
      virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

