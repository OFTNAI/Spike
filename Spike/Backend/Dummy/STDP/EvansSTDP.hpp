#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class EvansSTDP : public virtual ::Backend::Dummy::STDPCommon,
                      public ::Backend::EvansSTDP {
    public:
      virtual void reset_state() {}
    };
  }
}
