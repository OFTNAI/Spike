#pragma once

#include "Spike/STDP/STDP.hpp"

namespace Backend {
  namespace Dummy {
    class STDP : public virtual ::Backend::STDP {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
