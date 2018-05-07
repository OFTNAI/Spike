#pragma once

#include "Spike/Synapses/VoltageSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class VoltageSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapses,
                                       public virtual ::Backend::VoltageSpikingSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(VoltageSpikingSynapses);
      
      void prepare() override;
      void reset_state() override;
    };
  } // namespace Dummy
} // namespace Backend

