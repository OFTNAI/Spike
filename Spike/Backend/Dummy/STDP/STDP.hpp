#pragma once

#include "Spike/STDP/STDP.hpp"

namespace Backend {
  namespace Dummy {
    class STDPCommon : public virtual ::Backend::STDPCommon {
    public:
    };

    class STDP : public virtual ::Backend::Dummy::STDPCommon,
                 public ::Backend::STDP {
    public:
    };
  }
}
