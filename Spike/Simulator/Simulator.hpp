#ifndef Simulator_H
#define Simulator_H

// #define SILENCE_SIMULATOR_SETUP
// #define VERBOSE_SIMULATION

#include "Spike/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"
#include "Spike/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"
#include "Spike/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Models/SpikingModel.hpp"

#include "Spike/Helpers/TimerWithMessages.hpp"



struct Simulator_Recording_Electrodes_To_Use_Struct {

	Simulator_Recording_Electrodes_To_Use_Struct(): count_neuron_spikes_recording_electrodes_bool(false), 
													count_input_neuron_spikes_recording_electrodes_bool(false), 
													collect_neuron_spikes_recording_electrodes_bool(false), 
													collect_input_neuron_spikes_recording_electrodes_bool(false), 
													network_state_archive_recording_electrodes_bool(false), 
													collect_neuron_spikes_optional_parameters(new Collect_Neuron_Spikes_Optional_Parameters()),
													collect_input_neuron_spikes_optional_parameters(new Collect_Neuron_Spikes_Optional_Parameters()),
													network_state_archive_optional_parameters(new Network_State_Archive_Optional_Parameters()) 
													{}

	bool count_neuron_spikes_recording_electrodes_bool;
	bool count_input_neuron_spikes_recording_electrodes_bool;
	bool collect_neuron_spikes_recording_electrodes_bool;
	bool collect_input_neuron_spikes_recording_electrodes_bool;
	bool network_state_archive_recording_electrodes_bool;

	Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters;
	Collect_Neuron_Spikes_Optional_Parameters * collect_input_neuron_spikes_optional_parameters;
	Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters;

};


struct Simulator_Run_Simulation_General_Options {

	Simulator_Run_Simulation_General_Options(): presentation_time_per_stimulus_per_epoch(0.1), 
												number_of_epochs(1), 
												apply_stdp_to_relevant_synapses(false),
												stimulus_presentation_order_seed(1),
												reset_current_time_between_each_stimulus(false),
												reset_current_time_between_each_epoch(false),
												delete_spike_analyser_on_simulator_destruction(true),
												specific_epoch_to_pass_to_spike_analyser(0),
												reset_model_state_between_epochs(true)
												{}


	float presentation_time_per_stimulus_per_epoch; 
	int number_of_epochs;
	bool apply_stdp_to_relevant_synapses;
	int stimulus_presentation_order_seed;
	bool reset_current_time_between_each_stimulus;
	bool reset_current_time_between_each_epoch;
	bool delete_spike_analyser_on_simulator_destruction;
	int specific_epoch_to_pass_to_spike_analyser;
	bool reset_model_state_between_epochs;

};


struct Simulator_File_Storage_Options_Struct {

	Simulator_File_Storage_Options_Struct(): save_recorded_neuron_spikes_to_file(false), 
											save_recorded_input_neuron_spikes_to_file(false), 
											write_initial_synaptic_weights_to_file_bool(false), 
											human_readable_storage(false), 
											network_is_trained(false) 
											{}

	bool save_recorded_neuron_spikes_to_file;
	bool save_recorded_input_neuron_spikes_to_file;
	bool write_initial_synaptic_weights_to_file_bool;
	bool human_readable_storage;
	bool network_is_trained;

};

enum PRESENTATION_FORMAT {
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI,
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS,
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_NO_RESET,
		PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS,
		PRESENTATION_FORMAT_RANDOM_NO_RESET
	};

enum OBJECT_ORDER {
	OBJECT_ORDER_ORIGINAL,
	OBJECT_ORDER_RANDOM
};

enum TRANSFORM_ORDER {
	TRANSFORM_ORDER_ORIGINAL,
	TRANSFORM_ORDER_RANDOM
};

struct Stimuli_Presentation_Struct {

	Stimuli_Presentation_Struct(): presentation_format(PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI), 
								object_order(OBJECT_ORDER_ORIGINAL), 
								transform_order(TRANSFORM_ORDER_ORIGINAL), 
								reset_current_time_between_each_stimulus(false) 
								{}

	PRESENTATION_FORMAT presentation_format;
	OBJECT_ORDER object_order;
	TRANSFORM_ORDER transform_order;
	bool reset_current_time_between_each_stimulus;

};


struct Simulator_Options {
	Simulator_Options(): run_simulation_general_options(new Simulator_Run_Simulation_General_Options()), 
						recording_electrodes_options(new Simulator_Recording_Electrodes_To_Use_Struct()), 
						file_storage_options(new Simulator_File_Storage_Options_Struct()),
						stimuli_presentation_options(new Stimuli_Presentation_Struct())
						{}

	Simulator_Run_Simulation_General_Options * run_simulation_general_options;
	Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_options;
	Simulator_File_Storage_Options_Struct * file_storage_options;
	Stimuli_Presentation_Struct * stimuli_presentation_options;
};


// Simulator Class for running of the simulations
class Simulator{
public:
	// Constructor/Destructor
	Simulator(SpikingModel * spiking_model_param, Simulator_Options * simulator_options_param);
	~Simulator();


	// Variables
        Context* context = nullptr;
	std::string full_directory_name_for_simulation_data_files;
	int simulations_run_count;

	// Host Pointers
	SpikingModel * spiking_model = nullptr;
        SpikeAnalyser * spike_analyser = nullptr;
	Simulator_Options * simulator_options = nullptr;

	CountNeuronSpikesRecordingElectrodes* count_neuron_spikes_recording_electrodes = nullptr;
	CountNeuronSpikesRecordingElectrodes* count_input_neuron_spikes_recording_electrodes = nullptr;
	CollectNeuronSpikesRecordingElectrodes* collect_neuron_spikes_recording_electrodes = nullptr;
	CollectNeuronSpikesRecordingElectrodes* collect_input_neuron_spikes_recording_electrodes = nullptr;
	NetworkStateArchiveRecordingElectrodes* network_state_archive_recording_electrodes = nullptr;
	
	// Functions

	void CreateDirectoryForSimulationDataFiles(std::string directory_name_for_simulation_data_files);

	void reset_all_recording_electrodes();

	// void RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool collect_spikes, bool save_collected_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);
	// void RunSimulationToCollectEvents(float presentation_time_per_stimulus_per_epoch, bool isTrained);
	// void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed);
	void RunSimulation();


protected: 
	int* setup_stimuli_presentation_order();

	void perform_per_timestep_recording_electrode_instructions(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_stimulus_per_epoch, int epoch_number);
	void perform_pre_stimulus_presentation_instructions(int stimulus_index);
	void perform_post_stimulus_presentation_instructions(int epoch_number);
	void perform_post_epoch_instructions(int epoch_number, TimerWithMessages * epoch_timer);
	void perform_end_of_simulation_instructions(TimerWithMessages * simulation_timer);
};
#endif
