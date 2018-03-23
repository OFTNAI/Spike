#ifndef NetworkStateArchiveRecordingElectrodes_H
#define NetworkStateArchiveRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.hpp"

class NetworkStateArchiveRecordingElectrodes; // forward definition

namespace Backend {
  class NetworkStateArchiveRecordingElectrodes : public virtual RecordingElectrodes {
  public:
    SPIKE_ADD_BACKEND_FACTORY(NetworkStateArchiveRecordingElectrodes);
  };
}

struct Network_State_Archive_Optional_Parameters {

	Network_State_Archive_Optional_Parameters(): human_readable_storage(false), output_weights_only(false) {}

		bool human_readable_storage;
		bool output_weights_only;
	
};


class NetworkStateArchiveRecordingElectrodes  : public RecordingElectrodes {
public:
  SPIKE_ADD_BACKEND_GETSET(NetworkStateArchiveRecordingElectrodes,
                           RecordingElectrodes);
  void init_backend(Context* ctx = _global_ctx) override;
  ~NetworkStateArchiveRecordingElectrodes() = default;

  // Host Pointers
  Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters = nullptr;

  // Constructor/Destructor
  NetworkStateArchiveRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);

  void initialise_network_state_archive_recording_electrodes(Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters_param);

  void write_initial_synaptic_weights_to_file();
  void write_network_state_to_file();

private:
  std::shared_ptr<::Backend::NetworkStateArchiveRecordingElectrodes> _backend;
};

#endif
