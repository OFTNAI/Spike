// Vogels Abbot Benchmark Network
// Author: Nasir Ahmad (Created: 16/09/2016)
// make FILE='VogelsAbbotNet' EXPERIMENT_DIRECTORY='Experiments'  model -j8


/*
	This network has been created to benchmark Spike. It shall follow the network
	used to analyse Auryn.

	Publications:
	Vogels, Tim P., and L. F. Abbott. 2005. “Signal Propagation and Logic Gating in Networks of Integrate-and-Fire Neurons.” The Journal of Neuroscience: The Official Journal of the Society for Neuroscience 25 (46): 10786–95.
	Zenke, Friedemann, and Wulfram Gerstner. 2014. “Limits to High-Speed Simulations of Spiking Neural Networks Using General-Purpose Computers.” Frontiers in Neuroinformatics 8 (August). Frontiers. doi:10.3389/fninf.2014.00076.

*/


#include "Spike/Models/SpikingModel.hpp"
#include "Spike/Simulator/Simulator.hpp"
#ifdef SPIKE_WITH_CUDA
#include <cuda_profiler_api.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>

/*
	When running simulations with the AdEx neuron type, I need a number of parameters:
		membrane_capacitance_Cm
		membrane_leakage_conductance_g0
		leak_reversal_potential_E_Lo
		slope_factor_Delta_T
		adaptation_coupling_coefficient_a
		adaptation_time_constant_tau_w
		adaptation_change_b

	In order to simply this AdEx model to a LIF neuron, we can:
	(See: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
	-> Take the limit Delta_T => 0
		-> To do this I shall introduce an if so we can set Delta_T = 0
	-> Remove the adaptation variable w
		-> Set adaption coupling coefficient a = 0
		-> Set adaptation change on spike b = 0
		-> Adaptation time constant does not have an effect, tau_w = 1 (we take the inverse so cannot be zero)

*/


int main (int argc, char *argv[]){
#ifdef SPIKE_WITH_CUDA
	cudaProfilerStart();
#endif
	// TIMESTEP MUST BE SET BEFORE DATA IS IMPORTED. USED FOR ROUNDING.
	// The details below shall be used in a SpikingModel
	SpikingModel * BenchModel = new SpikingModel();
	float timestep = 0.0001f; // 50us for now
	BenchModel->SetTimestep(timestep);

	// Create neuron, synapse and stdp types for this model
	AdExSpikingNeurons * adex_spiking_neurons = new AdExSpikingNeurons();
	GeneratorInputSpikingNeurons * generator_input_spiking_neurons = new GeneratorInputSpikingNeurons();
	ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
	MasquelierSTDP * masquelier_stdp = new MasquelierSTDP();
	// No STDP implemented

	// Add my populations to the SpikingModel
	BenchModel->spiking_neurons = adex_spiking_neurons;
	BenchModel->input_spiking_neurons = generator_input_spiking_neurons;
	BenchModel->spiking_synapses = conductance_spiking_synapses;
	BenchModel->stdp_rule = masquelier_stdp;

	// Set up Neuron Parameters
	// AdEx
	AdEx_spiking_neuron_parameters_struct * EXC_NEURON_PARAMS = new AdEx_spiking_neuron_parameters_struct();
	AdEx_spiking_neuron_parameters_struct * INH_NEURON_PARAMS = new AdEx_spiking_neuron_parameters_struct();

	EXC_NEURON_PARAMS->membrane_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF
	INH_NEURON_PARAMS->membrane_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF

	EXC_NEURON_PARAMS->membrane_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS
	INH_NEURON_PARAMS->membrane_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS

	EXC_NEURON_PARAMS->leak_reversal_potential_E_L = -80.0f*pow(10.0, -3);
	INH_NEURON_PARAMS->leak_reversal_potential_E_L = -80.0f*pow(10.0, -3);

	EXC_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);	// -74mV
	INH_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);	// -82mV

	EXC_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms
	INH_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms

	EXC_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold
	INH_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold

	// EXC_NEURON_PARAMS->background_current = 0.000000030000365f*pow(10.0, -2); //
	// INH_NEURON_PARAMS->background_current = 0.000000030000365f*pow(10.0, -2); //
	// EXC_NEURON_PARAMS->background_current = 400.0f*pow(10.0, -12); //
	// INH_NEURON_PARAMS->background_current = 400.0f*pow(10.0, -12); //

	// Turning these off for normal LIF behaviour
	EXC_NEURON_PARAMS->slope_factor_Delta_T = 0.0f*pow(10.0, -3);  // mV
	INH_NEURON_PARAMS->slope_factor_Delta_T = 0.0f*pow(10.0, -3);  // mV

	EXC_NEURON_PARAMS->adaptation_coupling_coefficient_a = 0.0f*pow(10, -9);  // nS
	INH_NEURON_PARAMS->adaptation_coupling_coefficient_a = 0.0f*pow(10, -9);  // nS

	EXC_NEURON_PARAMS->adaptation_time_constant_tau_w = 1.0f;  // ms
	INH_NEURON_PARAMS->adaptation_time_constant_tau_w = 1.0f;  // ms

	EXC_NEURON_PARAMS->adaptation_change_b = 0.0f*pow(10, -9);  // nA
	INH_NEURON_PARAMS->adaptation_change_b = 0.0f*pow(10, -9);  // nA

	/*
		Set up STDP Parameters
	*/
	masquelier_stdp_parameters_struct * STDP_PARAMS = new masquelier_stdp_parameters_struct();
	STDP_PARAMS->a_minus = 0.001f;
	STDP_PARAMS->a_plus = 0.001f;
	STDP_PARAMS->tau_minus = 0.50f;
	STDP_PARAMS->tau_plus = 0.05f;
	masquelier_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) adex_spiking_neurons, (SpikingNeurons *) generator_input_spiking_neurons, (stdp_parameters_struct *) STDP_PARAMS);


	/*
		Setting up INPUT NEURONS
	*/
	// Creating an input neuron parameter structure
	generator_input_spiking_neuron_parameters_struct* input_neuron_params = new generator_input_spiking_neuron_parameters_struct();
	// Setting the dimensions of the input neuron layer
	input_neuron_params->group_shape[0] = 1;		// x-dimension of the input neuron layer
	input_neuron_params->group_shape[1] = 20;		// y-dimension of the input neuron layer
	// Create a group of input neurons. This function returns the ID of the input neuron group
	int input_layer_ID = BenchModel->AddInputNeuronGroup(input_neuron_params);



	/*
		Add STIMULUS
	*/
	std::vector<std::string> stimulifiles;
	stimulifiles.push_back("./DATA/VABenchmark");

	// Running through each foldername:
	for (int i=0; i < stimulifiles.size(); i++){
		std::ifstream stimulusNUMS((stimulifiles[i] + "_SpikesPerNeuron.csv"));
		std::ifstream stimulusSPIKES((stimulifiles[i] + "_SpikeTimes.csv"));
		std::vector<int> num_spikes;
		std::vector<float> spike_times_vec;
		int totalnumspikes = 0;

		// Get the number of spikes for this stimulus
		while( stimulusNUMS.good() )
		{
			string fileline;
			std::stringstream iss;
			getline( stimulusNUMS, fileline);
			if (stimulusNUMS.eof()) {
				stimulusNUMS.close();
				break;
			}
			iss << fileline;
			string token;
			while ( getline( iss, token, ','))
			{
				totalnumspikes += std::stoi(token);
				num_spikes.push_back( std::stoi(token) );
			}

		}
		std::cout << totalnumspikes << '\n';

		// Get the spike times for this stimulus
		while( stimulusSPIKES.good() )
		{
			string fileline;
			std::stringstream iss;
			getline( stimulusSPIKES, fileline);
			if (stimulusSPIKES.eof()) {
				stimulusSPIKES.close();
				break;
			}
			iss << fileline;
			string token;
			while ( getline( iss, token, ','))
			{
				spike_times_vec.push_back( roundf(std::stof(token) / timestep) * timestep );
			}
		}

		// Creating pointers for storage of my indices and times
		int* spike_ids = (int*)malloc(totalnumspikes*sizeof(int));
		float* spike_times = (float*)malloc(totalnumspikes*sizeof(float));
		// Assigning spike ids and times
		int index = 0;
		for (int i = 0; i < num_spikes.size(); i++){
			for (int j = 0; j < num_spikes[i]; j++){
				spike_ids[index] = i;
				spike_times[index] = spike_times_vec[index];
				index++;
			}
		}

		// Add my arrays to the simulation:
		generator_input_spiking_neurons->AddStimulus(totalnumspikes, spike_ids, spike_times);
	}





	/*
		Setting up NEURON POPULATION
	*/
	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;
	// Creating a single exc and inh population for now
	EXC_NEURON_PARAMS->group_shape[0] = 1;
	EXC_NEURON_PARAMS->group_shape[1] = 3200;
	INH_NEURON_PARAMS->group_shape[0] = 1;
	INH_NEURON_PARAMS->group_shape[1] = 800;
	EXCITATORY_NEURONS.push_back(BenchModel->AddNeuronGroup(EXC_NEURON_PARAMS));
	INHIBITORY_NEURONS.push_back(BenchModel->AddNeuronGroup(INH_NEURON_PARAMS));

	/*
		Setting up SYNAPSES
	*/
	conductance_spiking_synapse_parameters_struct * EXC_OUT_SYN_PARAMS = new conductance_spiking_synapse_parameters_struct();
	conductance_spiking_synapse_parameters_struct * INH_OUT_SYN_PARAMS = new conductance_spiking_synapse_parameters_struct();
	conductance_spiking_synapse_parameters_struct * INPUT_SYN_PARAMS = new conductance_spiking_synapse_parameters_struct();
	// Setting delays
	EXC_OUT_SYN_PARAMS->delay_range[0] = 0.0008; // 0.8ms
	EXC_OUT_SYN_PARAMS->delay_range[1] = 0.0008;
	INH_OUT_SYN_PARAMS->delay_range[0] = 0.0008;
	INH_OUT_SYN_PARAMS->delay_range[1] = 0.0008;
	INPUT_SYN_PARAMS->delay_range[0] = timestep;
	INPUT_SYN_PARAMS->delay_range[1] = timestep;
	// Setting Reversal Potentials for specific synapses (according to evans paper)
	EXC_OUT_SYN_PARAMS->reversal_potential_Vhat = 0.0f;
	INH_OUT_SYN_PARAMS->reversal_potential_Vhat = -80.0f;
	INPUT_SYN_PARAMS->reversal_potential_Vhat = 0.0f;
	// Set Weight Range?
	EXC_OUT_SYN_PARAMS->weight_range_bottom = 4.0f*10.0f*pow(10.0, -9);
	EXC_OUT_SYN_PARAMS->weight_range_top = 4.0f*10.0f*pow(10.0, -9);
	INH_OUT_SYN_PARAMS->weight_range_bottom = 51.0f*10.0f*pow(10.0, -9);
	INH_OUT_SYN_PARAMS->weight_range_top = 51.0f*10.0f*pow(10.0, -9);
	INPUT_SYN_PARAMS->weight_range_bottom = 100.0f;
	INPUT_SYN_PARAMS->weight_range_top = 100.0f;
	// Set timescales
	EXC_OUT_SYN_PARAMS->decay_term_tau_g = 5.0f*pow(10.0, -3);  // 5ms
	INH_OUT_SYN_PARAMS->decay_term_tau_g = 10.0f*pow(10.0, -3);  // 10ms
	INPUT_SYN_PARAMS->decay_term_tau_g = 1.0f*timestep;  // Smallest

	// Biological Scaling factors
	EXC_OUT_SYN_PARAMS->biological_conductance_scaling_constant_lambda = 1.0;
	INH_OUT_SYN_PARAMS->biological_conductance_scaling_constant_lambda = 1.0;

	// Creating Synapse Populations
	EXC_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	INH_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	INPUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	EXC_OUT_SYN_PARAMS->stdp_on = false;
	INH_OUT_SYN_PARAMS->stdp_on = false;
	INPUT_SYN_PARAMS->stdp_on = false;
	EXC_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%
	INH_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%
	INPUT_SYN_PARAMS->random_connectivity_probability = 0.01; // 1%
	// Connect all of the populations
	BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0], INH_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0], INH_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(input_layer_ID, EXCITATORY_NEURONS[0], INPUT_SYN_PARAMS);


	/*
		COMPLETE NETWORK SETUP
	*/
	BenchModel->high_fidelity_spike_storage = false;
	BenchModel->finalise_model();
	// BenchModel->copy_model_to_device();


	// Create the simulator options
	Simulator_Options* simoptions = new Simulator_Options();
	simoptions->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = 10.0f;
	simoptions->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;
	// simoptions->recording_electrodes_options->count_input_neuron_spikes_recording_electrodes_bool = true;
	// simoptions->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool = true;
	// simoptions->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool = true;
	// simoptions->file_storage_options->save_recorded_neuron_spikes_to_file = true;
	// simoptions->file_storage_options->human_readable_storage = true;

	Simulator * simulator = new Simulator(BenchModel, simoptions);
	simulator->RunSimulation();

#ifdef SPIKE_WITH_CUDA
	cudaProfilerStop();
#endif
}
