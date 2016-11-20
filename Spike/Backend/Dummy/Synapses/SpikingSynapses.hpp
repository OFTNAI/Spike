#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingSynapses : public ::Backend::SpikingSynapses {
    public:
      // virtual void prepare() {}
      virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

