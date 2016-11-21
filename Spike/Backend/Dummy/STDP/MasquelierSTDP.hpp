#pragma once

#include "Spike/STDP/MasquelierSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Dummy {
    class MasquelierSTDP : public virtual ::Backend::Dummy::STDPCommon,
                           public ::Backend::MasquelierSTDP {
    public:
      virtual void reset_state() {}
    };
  }
}
