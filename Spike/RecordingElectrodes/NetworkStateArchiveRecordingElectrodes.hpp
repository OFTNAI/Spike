#ifndef NetworkStateArchiveRecordingElectrodes_H
#define NetworkStateArchiveRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.hpp"

class NetworkStateArchiveRecordingElectrodes; // forward definition

namespace Backend {
  class NetworkStateArchiveRecordingElectrodes : public virtual RecordingElectrodes {
  public:
    ADD_FRONTEND_GETTER(NetworkStateArchiveRecordingElectrodes);

    virtual void prepare() {
      printf("TODO Backend::NetworkStateArchiveRecordingElectrodes::prepare\n");
    }

    virtual void copy_state_to_front(::NetworkStateArchiveRecordingElectrodes* front) = 0;
  };
}

#include "Spike/Backend/Dummy/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#endif


struct Network_State_Archive_Optional_Parameters {

	Network_State_Archive_Optional_Parameters(): human_readable_storage(false) {}

		bool human_readable_storage;
	
};


class NetworkStateArchiveRecordingElectrodes  : public RecordingElectrodes {
public:
  ADD_BACKEND_GETTER(NetworkStateArchiveRecordingElectrodes);
  virtual void prepare_backend(Context* ctx = _global_ctx);
  virtual void reset_state();

  // Host Pointers
  Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters = nullptr;

  // Constructor/Destructor
  NetworkStateArchiveRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);

  void initialise_network_state_archive_recording_electrodes(Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters_param);

  void write_initial_synaptic_weights_to_file();
  void write_network_state_to_file();
};

#endif
