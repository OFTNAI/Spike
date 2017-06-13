#pragma once

#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class ConductanceSpikingSynapses : public virtual ::Backend::Dummy::SpikingSynapses,
                                       public virtual ::Backend::ConductanceSpikingSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(ConductanceSpikingSynapses);
      
      void prepare() override;
      void reset_state() override;
    };
  } // namespace Dummy
} // namespace Backend

