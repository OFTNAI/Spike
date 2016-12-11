#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <string>
using namespace std;

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"

class RecordingElectrodes; // forward definition

namespace Backend {
  class RecordingElectrodes : public virtual SpikeBackendBase {
  public:
    ADD_FRONTEND_GETTER(RecordingElectrodes);
  };
}

#include "Spike/Backend/Dummy/RecordingElectrodes/RecordingElectrodes.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/RecordingElectrodes/RecordingElectrodes.hpp"
#endif

class RecordingElectrodes : public virtual SpikeBase {
public:
  RecordingElectrodes(SpikingNeurons * neurons_parameter,
                      SpikingSynapses * synapses_parameter,
                      string full_directory_name_for_simulation_data_files_param,
                      const char * prefix_string_param);

  ADD_BACKEND_GETTER(RecordingElectrodes);
  virtual void init_backend(Context* ctx = _global_ctx) = 0;
  virtual void reset_state() = 0;

  // Variables
  std::string full_directory_name_for_simulation_data_files;
  const char * prefix_string;

  // Host Pointers
  SpikingNeurons * neurons;
  SpikingSynapses * synapses;
};

#endif
