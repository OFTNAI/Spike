#ifndef Simulator_H
#define Simulator_H
// Silences the printfs
//#define QUIETSTART

// cuRand Library import
#include <curand.h>
#include <curand_kernel.h>

//	CUDA library
#include <cuda.h>

#include "../RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.h"
#include "../RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.h"
#include "../RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Models/SpikingModel.h"

#include "../Helpers/TimerWithMessages.h"

struct Simulator_Run_Simulation_General_Options {

	Simulator_Run_Simulation_General_Options(): presentation_time_per_stimulus_per_epoch(), number_of_epochs(1), apply_stdp_to_relevant_synapses(false), stimulus_presentation_order_seed(1) {}


	float presentation_time_per_stimulus_per_epoch; 
	int number_of_epochs;
	bool apply_stdp_to_relevant_synapses;
	int stimulus_presentation_order_seed;
	bool reset_current_time_between_each_stimulus;

};


struct Simulator_Recording_Electrodes_To_Use_Struct {

	Simulator_Recording_Electrodes_To_Use_Struct(): count_neuron_spikes_recording_electrodes_bool(false), count_input_neuron_spikes_recording_electrodes_bool(false), collect_neuron_spikes_recording_electrodes_bool(false), collect_input_neuron_spikes_recording_electrodes_bool(false), network_state_archive_recording_electrodes_bool(false)  {}

	bool count_neuron_spikes_recording_electrodes_bool;
	bool count_input_neuron_spikes_recording_electrodes_bool;
	bool collect_neuron_spikes_recording_electrodes_bool;
	bool collect_input_neuron_spikes_recording_electrodes_bool;
	bool network_state_archive_recording_electrodes_bool;

};


struct Simulator_File_Storage_Options_Struct {

	Simulator_File_Storage_Options_Struct(): save_recorded_neuron_spikes_to_file(false), save_recorded_input_neuron_spikes_to_file(false), write_initial_synaptic_weights_to_file_bool(false), human_readable_storage(false), network_is_trained(false) {}

	bool save_recorded_neuron_spikes_to_file;
	bool save_recorded_input_neuron_spikes_to_file;
	bool write_initial_synaptic_weights_to_file_bool;
	bool human_readable_storage;
	bool network_is_trained;

};


// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator();
	~Simulator();


	// Variables
	float timestep;
	std::string full_directory_name_for_simulation_data_files;
	bool high_fidelity_spike_storage; // Flag: Enable for high accuracy spike storage, Disable for speed
	int number_of_simulations_run;

	// Host Pointers
	SpikingModel * spiking_model;
	Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct;
	CountNeuronSpikesRecordingElectrodes* count_neuron_spikes_recording_electrodes;
	CountNeuronSpikesRecordingElectrodes* count_input_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* collect_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* collect_input_neuron_spikes_recording_electrodes;
	NetworkStateArchiveRecordingElectrodes* network_state_archive_recording_electrodes;
	
	// Functions
	void SetTimestep(float timest);
	void SetSpikingModel(SpikingModel * spiking_model_parameter);

	void CreateDirectoryForSimulationDataFiles(std::string directory_name_for_simulation_data_files);
	
	void prepare_recording_electrodes(Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct, Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters, Collect_Neuron_Spikes_Optional_Parameters * collect_input_neuron_spikes_optional_parameters, Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters);
	void reset_all_recording_electrodes();

	// void RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool collect_spikes, bool save_collected_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);
	// void RunSimulationToCollectEvents(float presentation_time_per_stimulus_per_epoch, bool isTrained);
	// void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed);
	void RunSimulation(Simulator_Run_Simulation_General_Options * simulator_run_simulation_general_options_struct, Stimuli_Presentation_Struct * stimuli_presentation_params, Simulator_File_Storage_Options_Struct * simulator_file_storage_options_struct, SpikeAnalyser *spike_analyser);


protected: 
	void perform_per_timestep_recording_electrode_instructions(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_stimulus_per_epoch);
	void perform_pre_stimulus_presentation_instructions(int stimulus_index, Stimuli_Presentation_Struct * stimuli_presentation_params);
	void perform_post_stimulus_presentation_instructions(SpikeAnalyser* spike_analyser);
	void perform_post_epoch_instructions(int epoch_number, TimerWithMessages * epoch_timer, Simulator_File_Storage_Options_Struct * simulator_file_storage_options_struct);
	void perform_end_of_simulation_instructions(TimerWithMessages * simulation_timer);
};
#endif
