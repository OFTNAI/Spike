#pragma once

#include "Spike/Monitors/Monitors.hpp"

namespace Backend {
  namespace Dummy {
    class Monitors : public virtual ::Backend::Monitors {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
