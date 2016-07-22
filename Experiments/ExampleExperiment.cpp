/*

	An Example Model for running the SPIKE simulator

	Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

	To create the executable for this network, run:
	make FILE='ExampleExperiment' EXPERIMENT_DIRECTORY='Experiments'  model -j8

	To create your own simulation, simply create a .cpp file similar to this one (with the network structure you desire) and run:
	make FILE='YOUREXPERIMENTFILENAME' EXPERIMENT_DIRECTORY='Experiments'  model -j8


*/



// The Simulator Class
#include "../Simulator/Simulator.h"							// The simulator class takes references to your neuron/synapse populations and runs the simulation

// Input Neuron Classes
#include "../Neurons/SpikingNeurons.h"						// The Spiking Neurons parent class is used when passing references to the simulator
#include "../Neurons/GeneratorInputSpikingNeurons.h"		// The Generator Input neuron type allows you to load and give an input neuron population specific spike times
// #include "../Neurons/PoissonInputSpikingNeurons.h"		// Poisson Input Neurons allow you to set the Poisson Rate of the population of input neurons

// Neuron Classes
#include "../Neurons/SpikingSynapses.h"						// The Spiking Synapses parent class is used when passing references to the simulator
#include "../Neurons/LIFSpikingNeurons.h"					// Leaky Integrate and Fire Implementation
// #include "../Neurons/IzhikevichSpikingNeurons.h"			// Izhikevich Spiking Neuron Implementation


// Synapse Classes
#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/CurrentSpikingSynapses.h"				// Current Spiking Synapses inject a square wave of current into a post-synaptic neuron when that synapse is active	
// #include "../Synapses/ConductanceSpikingSynapses.h"		// Conductance Spiking Synapses have a decaying conductance associated with synapses to inject current into post-synaptic neurons with some decay

// STDP Class
#include "../STDP/STDP.h"									// STDP class used to pass references to the simulator
#include "../STDP/Higgins.h"								// STDP rule used by Higgins in: http://biorxiv.org/content/early/2016/06/17/059428
// #include "../STDP/EvansSTDP.h"							// STDP rule used by Evans in: http://www.ncbi.nlm.nih.gov/pubmed/22848199

// Spike Analyser class for information analyses
#include "../SpikeAnalyser/SpikeAnalyser.h"

// Other helper code
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/TimerWithMessages.h"



// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	/*
			CHOOSE THE COMPONENTS OF YOUR SIMULATION
	*/

	// Create an instance of the Simulator
	Simulator simulator;

	// Set up the simulator with a timestep at which the neuron, synapse and STDP properties will be calculated 
	float timestep = 0.00002;  // In seconds
	simulator.SetTimestep(timestep);


	// Choose an input neuron type
	GeneratorInputSpikingNeurons* generator_input_neurons = new GeneratorInputSpikingNeurons();
	// PoissonInputSpikingNeurons* input neurons = new PoissonInputSpikingNeurons();

	// Choose your neuron type
	LIFSpikingNeurons* lif_spiking_neurons = new LIFSpikingNeurons();
	// IzhikevichSpikingNeurons* izh_spiking_neurons = new IzhikevichSpikingNeurons();

	// Choose your synapse type
	CurrentSpikingSynapses * current_spiking_synapses = new CurrentSpikingSynapses();
	// ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();

	// Choose an STDP type
	HigginsSTDP* higgins_stdp = new HigginsSTDP();
	// EvansSTDP * evans_stdp = new EvansSTDP();

	// Allocate your chosen components to the simulator
	simulator.SetInputNeuronType(generator_input_neurons);
	simulator.SetNeuronType(lif_spiking_neurons);
	simulator.SetSynapseType(current_spiking_synapses);
	simulator.SetSTDPType(higgins_stdp);


	/*
			SETUP PROPERTIES AND CREATE NETWORK:
		
		Note: 
		All Neuron, Synapse and STDP types have associated parameters structures.
		These structures are defined in the header file for that class and allow us to set properties.
	*/

	// SETTING UP INPUT NEURONS
	// Creating an input neuron parameter structure
	generator_input_spiking_neuron_parameters_struct* input_neuron_params = new generator_input_spiking_neuron_parameters_struct();
	// Setting the dimensions of the input neuron layer
	input_neuron_params->group_shape[0] = 1;		// x-dimension of the input neuron layer
	input_neuron_params->group_shape[1] = 10;		// y-dimension of the input neuron layer
	// Create a group of input neurons. This function returns the ID of the input neuron group
	int input_layer_ID = input_neurons.AddGroup(input_neuron_params);

	// We can now assign a set of spike times to neurons in the input layer
	int num_spikes = 5;
	int neuron_ids[5] = {0, 1, 3, 6, 7};
	float spike_times[5] = {0.1f, 0.3f, 0.2f, 0.5f, 0.9f};
	// Adding this stimulus to the input neurons
	generator_input_neurons->AddStimulus(num_spikes, neuron_ids, spike_times);


	// SETTING UP NEURON GROUPS
	// Creating an LIF parameter structure for an excitatory neuron population and an inhibitory
	// 1 x 100 Layer
	lif_spiking_neuron_parameters_struct * excitatory_population_params = new lif_spiking_neuron_parameters_struct();
	excitatory_population_params->group_shape[0] = 1;
	excitatory_population_params->group_shape[1] = 100;
	excitatory_population_params->resting_potential_v0 = -0.074f;
	excitatory_population_params->threshold_for_action_potential_spike = -0.053f;
	excitatory_population_params->somatic_capcitance_Cm = 500.0*pow(10, -12);
	excitatory_population_params->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);

	lif_spiking_neuron_parameters_struct * inhibitory_population_params = new lif_spiking_neuron_parameters_struct();
	inhibitory_population_params->group_shape[0] = 1;
	inhibitory_population_params->group_shape[1] = 25;
	inhibitory_population_params->resting_potential_v0 = -0.082f;
	inhibitory_population_params->threshold_for_action_potential_spike = -0.053f;
	inhibitory_population_params->somatic_capcitance_Cm = 214.0*pow(10, -12);
	inhibitory_population_params->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);

	// Create populations of excitatory and inhibitory neurons
	excitatory_neuron_layer_ID = simulator.AddNeuronGroup(excitatory_population_params);
	inhibitory_neuron_layer_ID = simulator.AddNeuronGroup(inhibitory_population_params);


	// SETTING UP SYNAPSES
	// Creating a synapses parameter structure for connections from the input neurons to the excitatory neurons
	current_spiking_synapse_parameters_struct* input_to_excitatory_parameters = new current_spiking_synapse_parameters_struct();
	input_to_excitatory_parameters->weight_range_bottom = 0.5f;		// Create uniform distributions of weights [0.5, 10.0]
	input_to_excitatory_parameters->weight_range_top = 10.0f;
	input_to_excitatory_parameters->delay_range[0] = timestep;		// Create uniform distributions of delays [1 timestep, 5 timesteps]
	input_to_excitatory_parameters->delay_range[1] = 5*timestep;
	// The connectivity types for synapses include:
		// CONNECTIVITY_TYPE_ALL_TO_ALL
		// CONNECTIVITY_TYPE_ONE_TO_ONE
		// CONNECTIVITY_TYPE_RANDOM
		// CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE
		// CONNECTIVITY_TYPE_SINGLE
	input_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	input_to_excitatory_parameters->stdp_on = true;

	// Creating a set of synapse parameters for connections from the excitatory neurons to the inhibitory neurons
	current_spiking_synapse_parameters_struct * excitatory_to_inhibitory_parameters = new current_spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = 10.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = 10.0f;
	excitatory_to_inhibitory_parameters->delay_range[0] = 5.0*timestep;
	excitatory_to_inhibitory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	excitatory_to_inhibitory_parameters->connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
	excitatory_to_inhibitory_parameters->stdp_on = false;

	// Creating a set of synapse parameters from the inhibitory neurons to the excitatory neurons
	conductance_spiking_synapse_parameters_struct * inhibitory_to_excitatory_parameters = new current_spiking_synapse_parameters_struct();
	excitatory_to_inhibitory_parameters->weight_range_bottom = -10.0f;
	excitatory_to_inhibitory_parameters->weight_range_top = -5.0f;
	inhibitory_to_excitatory_parameters->delay_range[0] = 5.0*timestep;
	inhibitory_to_excitatory_parameters->delay_range[1] = 3.0f*pow(10, -3);
	inhibitory_to_excitatory_parameters->connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	inhibitory_to_excitatory_parameters->stdp_on = false;
	

	// CREATING SYNAPSES
	// When creating synapses, the ids of the presynaptic and postsynaptic populations are all that are required
	// Note: Input neuron populations cannot be post-synaptic on any synapse
	simulator.AddSynapseGroup(input_layer_ID, excitatory_neuron_layer_ID, input_to_excitatory_parameters);
	simulator.AddSynapseGroup(excitatory_neuron_layer_ID, inhibitory_neuron_layer_ID, excitatory_to_inhibitory_parameters);
	simulator.AddSynapseGroup(inhibitory_neuron_layer_ID, excitatory_neuron_layer_ID, inhibitory_to_excitatory_parameters);

	// SETTING UP STDP
	// Getting the STDP parameter structure for this STDP type
	higgins_stdp_parameters_struct * STDP_PARAMS = new higgins_stdp_parameters_struct();	// You can use the default Values
	higgins_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) izhikevich_spiking_neurons, (SpikingNeurons *) input_neurons, (stdp_parameters_struct *) STDP_PARAMS);
	// evans_stdp_parameters_struct * STDP_PARAMS = new evans_stdp_parameters_struct(); 	// Or Define the parameters of the STDP model
	// STDP_PARAMS->decay_term_tau_C = 0.015;
	// STDP_PARAMS->decay_term_tau_D = 0.025;
	// evans_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) input_neurons, (stdp_parameters_struct *) STDP_PARAMS);



	
	/*
			SETUP THE SIMULATOR
	*/
	// This command concludes network creation and sets up variables required for the simulation
	simulator.setup_network();

	// If you wish to save the neuron spike times to a file, you must set-up a recording electrode for the neurons.
	// The recording electrode takes some parameters to ensure that it is able to record spikes before the GPU runs out of space
	// The values chosen below are on the safe side (but naturally therefore consume a relatively large amount of memory)
	/////////// SETUP RECORDING ELECTRODES ///////////
	int number_of_timesteps_per_device_spike_copy_check = 50;			// The number of timesteps after which the number of spikes on the GPU are checked
	int device_spike_store_size_multiple_of_total_neurons = 50;			// Defines the amount of allocated space on the GPU for spike storage
	float proportion_of_device_spike_store_full_before_copy = 0.2;		// The proportion of the memory on the GPU which has to be filled for the Recording electrode to copy the data off the GPU and clear it
	// Recording electrodes must be set for both the neuron and input neuron classes.
	simulator.setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator.setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);

	// A Struct (Present in the InputSpikingNeuron header) indicates with what style the stimuli should be shown to the network
	Stimuli_Presentation_Struct* stimuli_presentation_parameters = new Stimuli_Presentation_Struct();
	// If you wish to present with a random order: Reset_between_each_stimulus refers to a reset of the network state (i.e. membrane potential, conductances etc.)
	stimuli_presentation_parameters->PRESENTATION_FORMAT = PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS;

	/*
			RUN A SIMULATION
	*/
	// Before running the simulation, you must indicate
	// Run the simulation!
	simulator.RunSimulation(
		0.1f,								// Presentation time per stimulus
		1, 									// Number of Epochs to run the network for
		true,								// flag indicating whether the spike times should be recorded
		true,								// flag indicating whether the spike times should be saved to an outputs folder
		true,								// flag indicating whether STDP should be active for this simulation (false when testing the network)
		true,								// flag indicating whether the number of spikes for each neuron should be counted and displayed
		stimuli_presentation_parameters,	// Pointer to the Stimuli Presentation Struct	
		0,									// A seed for the random number generator which defines stimulus order
		NULL)								// If using visual stimuli (ImagePoissonInputSpikingNeuron type), you can set up the spike_analyser and reference it here. Other input types are note yet supported.

	return 0;
}
//