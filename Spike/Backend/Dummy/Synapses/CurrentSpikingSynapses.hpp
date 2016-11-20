#pragma once

#include "Spike/Synapses/CurrentSpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class CurrentSpikingSynapses : public ::Backend::CurrentSpikingSynapses {
    public:
      // virtual void prepare() {}
      // virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

