#pragma once

#include "Spike/ActivityMonitor/ActivityMonitor.hpp"

namespace Backend {
  namespace Dummy {
    class ActivityMonitor : public virtual ::Backend::ActivityMonitor {
    public:
      void prepare() override;
      void reset_state() override;
    };
  }
}
