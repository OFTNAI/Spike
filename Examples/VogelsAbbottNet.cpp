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
#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/Plasticity/VogelsSTDPPlasticity.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <getopt.h>
#include <time.h>
#include <iomanip>
#include <vector>

void connect_from_mat(
		int layer1,
		int layer2,
		conductance_spiking_synapse_parameters_struct* SYN_PARAMS, 
		std::string filename,
		SpikingModel* Model){

	ifstream weightfile;
	string line;
	stringstream ss;
	std::vector<int> prevec, postvec;
	int pre, post;
	float weight;
	int linecount = 0;
	weightfile.open(filename.c_str());

	if (weightfile.is_open()){
		printf("Loading weights from mat file: %s\n", filename.c_str());
		while (getline(weightfile, line)){
			if (line.c_str()[0] == '%'){
				continue;
			} else {
				linecount++;
				if (linecount == 1) continue;
				//printf("%s\n", line.c_str());
				ss.clear();
				ss << line;
				ss >> pre >> post >> weight;
				prevec.push_back(pre - 1);
				postvec.push_back(post - 1);
				//printf("%d, %d\n", pre, post);
			}
		}
		SYN_PARAMS->pairwise_connect_presynaptic = prevec;
		SYN_PARAMS->pairwise_connect_postsynaptic = postvec;
		SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_PAIRWISE;
		Model->AddSynapseGroup(layer1, layer2, SYN_PARAMS);
	}
}

int main (int argc, char *argv[]){
	// Getting options:
	float simtime = 20.0;
	bool fast = false;
	int num_timesteps_min_delay = 1;
	int num_timesteps_max_delay = 1;
	const char* const short_opts = "";
	const option long_opts[] = {
		{"simtime", 1, nullptr, 0},
		{"fast", 0, nullptr, 1},
		{"num_timesteps_min_delay", 1, nullptr, 2},
		{"num_timesteps_max_delay", 1, nullptr, 3}
	};
	// Check the set of options
	while (true) {
		const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

		// If none
		if (-1 == opt) break;

		switch (opt){
			case 0:
				printf("Running with a simulation time of: %ss\n", optarg);
				simtime = std::stof(optarg);
				break;
			case 1:
				printf("Running in fast mode (no spike collection)\n");
				fast = true;
				break;
			case 2:
				printf("Running with minimum delay: %s timesteps\n", optarg);
				num_timesteps_min_delay = std::stoi(optarg);
				if (num_timesteps_max_delay < num_timesteps_min_delay)
					num_timesteps_max_delay = num_timesteps_min_delay;
				break;
			case 3:
				printf("Running with maximum delay: %s timesteps\n", optarg);
				num_timesteps_max_delay = std::stoi(optarg);
				if (num_timesteps_max_delay < num_timesteps_min_delay){
					std::cerr << "ERROR: Max timestep shouldn't be smaller than min!" << endl;
					exit(1);
				}	
				break;
		}
	};
	
	// TIMESTEP MUST BE SET BEFORE DATA IS IMPORTED. USED FOR ROUNDING.
	// The details below shall be used in a SpikingModel
	SpikingModel * BenchModel = new SpikingModel();
	float timestep = 0.0001f; // 50us for now
	BenchModel->SetTimestep(timestep);

	// Create neuron, synapse and stdp types for this model
	LIFSpikingNeurons * lif_spiking_neurons = new LIFSpikingNeurons();
	PoissonInputSpikingNeurons * poisson_input_spiking_neurons = new PoissonInputSpikingNeurons();
	ConductanceSpikingSynapses * conductance_spiking_synapses = new ConductanceSpikingSynapses();
	// Add my populations to the SpikingModel
	BenchModel->spiking_neurons = lif_spiking_neurons;
	BenchModel->input_spiking_neurons = poisson_input_spiking_neurons;
	BenchModel->spiking_synapses = conductance_spiking_synapses;

	// Set up Neuron Parameters
	lif_spiking_neuron_parameters_struct * EXC_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();
	lif_spiking_neuron_parameters_struct * INH_NEURON_PARAMS = new lif_spiking_neuron_parameters_struct();

	EXC_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF
	INH_NEURON_PARAMS->somatic_capacitance_Cm = 200.0f*pow(10.0, -12);  // pF

	EXC_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS
	INH_NEURON_PARAMS->somatic_leakage_conductance_g0 = 10.0f*pow(10.0, -9);  // nS

	EXC_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);	// -74mV
	INH_NEURON_PARAMS->resting_potential_v0 = -60.0f*pow(10.0, -3);	// -82mV

	EXC_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms
	INH_NEURON_PARAMS->absolute_refractory_period = 5.0f*pow(10, -3);  // ms

	EXC_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold
	INH_NEURON_PARAMS->threshold_for_action_potential_spike = -50.0f*pow(10.0, -3); // -53mV threshold

	EXC_NEURON_PARAMS->background_current = 2.0f*pow(10.0, -2); //
	INH_NEURON_PARAMS->background_current = 2.0f*pow(10.0, -2); //


	/*
		Setting up INPUT NEURONS
	*/
	// Creating an input neuron parameter structure
	poisson_input_spiking_neuron_parameters_struct* input_neuron_params = new poisson_input_spiking_neuron_parameters_struct();
	// Setting the dimensions of the input neuron layer
	input_neuron_params->group_shape[0] = 1;		// x-dimension of the input neuron layer
	input_neuron_params->group_shape[1] = 20;		// y-dimension of the input neuron layer
	input_neuron_params->rate = 0.0f; // Hz
	// Create a group of input neurons. This function returns the ID of the input neuron group
	//int input_layer_ID = BenchModel->AddInputNeuronGroup(input_neuron_params);
	//poisson_input_spiking_neurons->set_up_rates();
	
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
	EXC_OUT_SYN_PARAMS->delay_range[0] = num_timesteps_min_delay*timestep;
	EXC_OUT_SYN_PARAMS->delay_range[1] = num_timesteps_max_delay*timestep;
	INH_OUT_SYN_PARAMS->delay_range[0] = num_timesteps_min_delay*timestep;
	INH_OUT_SYN_PARAMS->delay_range[1] = num_timesteps_max_delay*timestep;
	INPUT_SYN_PARAMS->delay_range[0] = num_timesteps_min_delay*timestep;
	INPUT_SYN_PARAMS->delay_range[1] = num_timesteps_max_delay*timestep;
	// Setting Reversal Potentials for specific synapses (according to evans paper)
	EXC_OUT_SYN_PARAMS->reversal_potential_Vhat = 0.0f*pow(10.0, -3);
	INH_OUT_SYN_PARAMS->reversal_potential_Vhat = -80.0f*pow(10.0, -3);
	INPUT_SYN_PARAMS->reversal_potential_Vhat = 0.0f*pow(10.0, -3);
	// Set Weight Range?
	EXC_OUT_SYN_PARAMS->weight_range_bottom = 0.4f;
	EXC_OUT_SYN_PARAMS->weight_range_top = 0.4f;
	INH_OUT_SYN_PARAMS->weight_range_bottom = 5.1f;
	INH_OUT_SYN_PARAMS->weight_range_top = 5.1f;
	INPUT_SYN_PARAMS->weight_range_bottom = 0.4f;
	INPUT_SYN_PARAMS->weight_range_top = 0.4f;
	// Set timescales
	EXC_OUT_SYN_PARAMS->decay_term_tau_g = 5.0f*pow(10.0, -3);  // 5ms
	INH_OUT_SYN_PARAMS->decay_term_tau_g = 10.0f*pow(10.0, -3);  // 10ms
	INPUT_SYN_PARAMS->decay_term_tau_g = 5.0f*pow(10.0, -3);

	// Biological Scaling factors
	EXC_OUT_SYN_PARAMS->biological_conductance_scaling_constant_lambda = 10.0f*pow(10.0,-9);
	INH_OUT_SYN_PARAMS->biological_conductance_scaling_constant_lambda = 10.0f*pow(10.0,-9);
	INPUT_SYN_PARAMS->biological_conductance_scaling_constant_lambda = 10.0f*pow(10.0,-9);

	/*
	// Creating Synapse Populations
	EXC_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	INH_OUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	INPUT_SYN_PARAMS->connectivity_type = CONNECTIVITY_TYPE_RANDOM;
	EXC_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
	INH_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
	INPUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
	EXC_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%
	INH_OUT_SYN_PARAMS->random_connectivity_probability = 0.02; // 2%
	INPUT_SYN_PARAMS->random_connectivity_probability = 0.01; // 1%

	// Connect all of the populations
	BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0], EXC_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0], INH_OUT_SYN_PARAMS);
	BenchModel->AddSynapseGroup(INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0], INH_OUT_SYN_PARAMS);
	//BenchModel->AddSynapseGroup(input_layer_ID, EXCITATORY_NEURONS[0], INPUT_SYN_PARAMS);
	*/
	EXC_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);
	INH_OUT_SYN_PARAMS->plasticity_vec.push_back(nullptr);

	connect_from_mat(
		EXCITATORY_NEURONS[0], EXCITATORY_NEURONS[0],
		EXC_OUT_SYN_PARAMS, 
		"../../../ee.wmat",
		BenchModel);
	connect_from_mat(
		EXCITATORY_NEURONS[0], INHIBITORY_NEURONS[0],
		EXC_OUT_SYN_PARAMS, 
		"../../../ei.wmat",
		BenchModel);
	connect_from_mat(
		INHIBITORY_NEURONS[0], EXCITATORY_NEURONS[0],
		INH_OUT_SYN_PARAMS, 
		"../../../ie.wmat",
		BenchModel);
	connect_from_mat(
		INHIBITORY_NEURONS[0], INHIBITORY_NEURONS[0],
		INH_OUT_SYN_PARAMS, 
		"../../../ii.wmat",
		BenchModel);


	// Adding connections based upon matrices given


	/*
		COMPLETE NETWORK SETUP
	*/
	BenchModel->finalise_model();


	// Create the simulator options
	Simulator_Options* simoptions = new Simulator_Options();
	simoptions->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = simtime;
	if (!fast){
		simoptions->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool = true;
		simoptions->file_storage_options->save_recorded_neuron_spikes_to_file = true;
		//simoptions->recording_electrodes_options->collect_neuron_spikes_optional_parameters->human_readable_storage = true;
		simoptions->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool = true;
	}


	Simulator * simulator = new Simulator(BenchModel, simoptions);
	clock_t starttime = clock();
	simulator->RunSimulation();
	clock_t totaltime = clock() - starttime;
	if ( fast ){
		std::ofstream timefile;
		timefile.open("timefile.dat");
		timefile << std::setprecision(10) << ((float)totaltime / CLOCKS_PER_SEC);
		timefile.close();
	}
	return(0);
}
