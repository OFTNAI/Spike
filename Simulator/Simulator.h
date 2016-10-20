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
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Models/SpikingModel.h"


struct Simulator_Recroding_Electrodes_To_Use_Struct {

	bool count_neuron_spikes_recording_electrodes_bool;
	bool input_count_neuron_spikes_recording_electrodes_bool;
	bool collect_neuron_spikes_recording_electrodes_bool;
	bool input_collect_neuron_spikes_recording_electrodes_bool;

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

	// Host Pointers
	SpikingModel * spiking_model;
	CountNeuronSpikesRecordingElectrodes* count_neuron_spikes_recording_electrodes;
	CountNeuronSpikesRecordingElectrodes* input_count_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* collect_neuron_spikes_recording_electrodes;
	CollectNeuronSpikesRecordingElectrodes* input_collect_neuron_spikes_recording_electrodes;

	
	// Functions
	void SetTimestep(float timest);
	void SetSpikingModel(SpikingModel * spiking_model_parameter);

	void CreateDirectoryForSimulationDataFiles(std::string directory_name_for_simulation_data_files);
	
	void prepare_recording_electrodes(Simulator_Recroding_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct);

	void RunSimulation(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron_for_single_cell_analysis, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed, SpikeAnalyser *spike_analyser,bool human_readable_storage, bool isTrained);
	void RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained);
	void RunSimulationToCollectEvents(float presentation_time_per_stimulus_per_epoch, bool isTrained);
	void RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed);


protected: 
	void per_timestep_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses);
};
#endif
