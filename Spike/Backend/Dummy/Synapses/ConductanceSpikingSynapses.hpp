#pragma once

#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class ConductanceSpikingSynapses : public ::Backend::ConductanceSpikingSynapses {
    public:
      // virtual void prepare() {}
      virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

