// An Example Model for running the SPIKE simulator
//
// Authors: Nasir Ahmad (16/03/2016), James Isbister (23/3/2016)

// To create the executable for this network, run:
// make FILE='ConductanceExperiment1' EXPERIMENT_DIRECTORY='Experiments'  model -j8


#include "Spike/Models/SpikingModel.hpp"
#include "Spike/Simulator/Simulator.hpp"
#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
#include "Spike/STDP/STDP.hpp"
#include "Spike/STDP/EvansSTDP.hpp"
#include "Spike/Neurons/Neurons.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Neurons/LIFSpikingNeurons.hpp"
#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"
#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Helpers/TimerWithMessages.hpp"
#include "Spike/Helpers/RandomStateManager.hpp"
#include <string>
#include <fstream>
//#include "Spike/Plotting/Plotter.hpp" // NB Not C++11 compatible
#include <vector>
#include <fstream>

#include <iostream>
using namespace std;

// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	TimerWithMessages experiment_timer;
	const int OPTIM_BIO_CONST_LAT = 1;
	const int OPTIM_BIO_CONST_FF = 2;
	const int OPTIM_BIO_CONST_LAT_FF = 3;
	const int OPTIM_FANINRAD = 4;
	const int OPTIM_DECAY = 5;
	const int OPTIM_STDP = 6;
	const int OPTIM_STDP_TAU = 7;
	const int OPTIM_FANINRAD_AND_SYNDECAY = 8;
	const int OPTIM_FF = 9;
	const int OPTIM_E2E_LAT = 10;
	const int OPTIM_E2E_FB = 11;
	const int OPTIM_DECAY_AND_BIO_CONST = 12;
	const int OPTIM_LAT = 13;
	const int OPTIM_E2E = 14;
	const int OBJFUNC_AVGFR = 1;
	const int OBJFUNC_MAXFR = 2;
	const int OBJFUNC_MAXINFO = 3;
	const int OBJFUNC_AVGINFO = 4;
	const int OBJFUNC_AVGFR_AND_MAXINFO = 5;
	const int OBJFUNC_AVGFR_AND_AVGINFO = 6;
	const int OBJFUNC_MAXFR_AND_AVGINFO = 7;
	const int OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY = 8;
	const int OBJFUNC_MAXFR_AND_MAXINFO_TRAINEDONLY = 9;
	const int OBJFUNC_AVGINFO_TRAINEDONLY = 10;
	const int OBJFUNC_MAXINFO_TRAINEDONLY = 11;
	const int OBJFUNC_AVGINFO_INPUT_TRAINEDONLY = 12;



	// Parameters related to Dakota Optimization
	int optimizationType = OPTIM_E2E_LAT; //OPTIM_BIO_CONST_LAT, OPTIM_BIO_CONST_FF, OPTIM_BIO_CONST_LAT_FF, OPTIM_FANINRAD, OPTIM_DECAY
	int objective_function = OBJFUNC_MAXINFO; //OBJFUNC_AVGFR, OBJFUNC_MAXFR, OBJFUNC_INFO, OBJFUNC_AVGFR_AND_INFO
	float optimal_average_firing_rate = 10.0f;//set if optimizing based on avgfr : Spontaneous rate (spikes/sec) 4.9 +- 7.1 (*1)
	const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	//*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.

	// Simulator Parameters
	string experimentName = "6.3grayAndBlack_a--StatDec_FFLATFB_20EP_testBO_FB5_rand123";
	// string inputs_for_test_name = "Inputs_test_multi";
	string inputs_for_test_name = "Inputs_test_BO";
	string inputs_for_train_name = "Inputs_train_BO";

	//try changing the order of presentation, original FF const, presentationTimeAtTraining x2 (0.2), 50ep
	float timestep = 0.00002;
	bool simulate_network_to_test_untrained = true;
	bool simulate_network_to_train_network = true;
	bool simulate_network_to_test_trained = true;
	bool human_readable_storage = false;
	bool plotInfoAnalysis = true;
	bool writeInformation = true;

	bool E2E_FB_ON = true;
	bool E2E_L_ON = true;
	bool E2E_L_STDP_ON = true;

	// Network Parameters
	const int number_of_layers = 4;
	int max_number_of_connections_per_pair = 2;
	int dim_excit_layer = 64;
	int dim_inhib_layer = 32;

	int fanInCount_G2E_FF = 30;
	int fanInCount_E2E_FF = 100;
	int fanInCount_E2I_L = 30;
	int fanInCount_I2E_L = 30;
	int fanInCount_E2E_L = 30;
	int fanInCount_E2E_FB = 5;//10;//10;


	if (fanInCount_E2E_FF%max_number_of_connections_per_pair!=0){
		printf("total_number_of_new_synapses has to be a multiple of max_number_of_connections_per_pair");
		return 0;
	}

	float gaussian_synapses_standard_deviation_G2E_FF = 12.0;//1.0;//12.0;
//	float gaussian_synapses_standard_deviation_E2E_FF = 8.0; //15.0;//10.0;//28.166444920;//10.0;//9.3631908834;//5.0;
	float gaussian_synapses_standard_deviation_E2E_FF[number_of_layers-1] = {12.0,18.0,24.0};//{8.0,12.0,16.0};//{6.0,9.0,12.0};//{8.0,12.0,16.0};//{12.0,18.0,18.0};
	float gaussian_synapses_standard_deviation_E2I_L = 1.0;
	float gaussian_synapses_standard_deviation_I2E_L = 8.0;
	float gaussian_synapses_standard_deviation_E2E_L = 8.0;
	float gaussian_synapses_standard_deviation_E2E_FB = 12.0;//8.0;

	float biological_conductance_scaling_constant_lambda_G2E_FF = 0.0001;
	float biological_conductance_scaling_constant_lambda_E2E_FF = 0.00005;//0.0003;//0.00008;
	float biological_conductance_scaling_constant_lambda_E2I_L = 0.001;
	float biological_conductance_scaling_constant_lambda_I2E_L = 0.005;//0.003;//0.02;//0.004;
	float biological_conductance_scaling_constant_lambda_E2E_L = 0.0001;//0.0001;//0.00008;
	float biological_conductance_scaling_constant_lambda_E2E_FB = 0.0001;//0.00008;

	float decay_term_tau_g_G2E_FF = 0.15;
	float decay_term_tau_g_E2E_FF = 0.15;//0.15;//0.15;//0.002 v. 0.15 and 0.15 is better?
	float decay_term_tau_g_E2E_L = 0.15;//0.002 v. 0.15 and 0.15 is better?
	float decay_term_tau_g_E2E_FB = 0.15;

	float decay_term_tau_g_E2I_L = 0.002;
	float decay_term_tau_g_I2E_L = 0.025;//0.005;//In Ben's model, 0.005 v 0.025 and latter produced better result


	// Neuronal Parameters
	float max_FR_of_input_Gabor = 100.0f;
	float absolute_refractory_period = 0.002;

	//Synaptic Parameters
	float weight_range_bottom = 0.0;
	float weight_range_top = 1.0;
	float learning_rate_rho = 0.1/timestep*100;//100.0;// 0.1;
	float decay_term_tau_C = 0.005;//0.3(In Ben's model, tau_C/tau_D = 0.003/0.005 v 0.015/0.025 v 0.075/0.125, and the first one produces the best result)
	float decay_term_tau_D = 0.005;

	float E2E_FF_minDelay = 5.0*timestep;
	float E2E_FF_maxDelay = 0.01;//3.0f*pow(10, -3);
	float E2I_L_minDelay = 5.0*timestep;
	float E2I_L_maxDelay = 0.01;//3.0f*pow(10, -3);
	float I2E_L_minDelay = 5.0*timestep;
	float I2E_L_maxDelay = 0.01;//3.0f*pow(10, -3);
	float E2E_FB_minDelay = 5.0*timestep;
	float E2E_FB_maxDelay = 0.01;
	float E2E_L_minDelay = 5.0*timestep;
	float E2E_L_maxDelay = 0.01;

	// Parameters for testing
	const float presentation_time_per_stimulus_per_epoch_test = 2.0f;
	bool record_spikes_test = true;
	bool save_recorded_spikes_and_states_to_file_test = true;

	// Parameters for training
	float presentation_time_per_stimulus_per_epoch_train = 0.2;//2.0f;
	int number_of_epochs_train = 20;//20;//10;

	// Parameters for Information Analysis
	int number_of_bins = 3;//5;
	bool useThresholdForMaxFR = true;//true;
	float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;//Max FR for info analysis
	// float max_firing_rate = 50.0*presentation_time_per_stimulus_per_epoch_test;//Max FR for info analysis


	// init parameters
	bool is_optimisation = false;
	bool isTrained=false;


	if (argc > 1) {
		is_optimisation = true;
		save_recorded_spikes_and_states_to_file_test = false;
		plotInfoAnalysis = false;

		switch (optimizationType){
			case OPTIM_BIO_CONST_LAT:
				biological_conductance_scaling_constant_lambda_E2I_L = stof(argv[4]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				biological_conductance_scaling_constant_lambda_I2E_L= stof(argv[5]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				break;
			case OPTIM_BIO_CONST_FF:
				biological_conductance_scaling_constant_lambda_E2E_FF= stof(argv[4]); //E2E_FF
				break;
			case OPTIM_BIO_CONST_LAT_FF:
				biological_conductance_scaling_constant_lambda_E2E_FF= stof(argv[4]); //E2E_FF
				biological_conductance_scaling_constant_lambda_E2I_L = stof(argv[5]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				biological_conductance_scaling_constant_lambda_I2E_L= stof(argv[6]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				break;
			case OPTIM_FANINRAD:
//				gaussian_synapses_standard_deviation_G2E_FF = stof(argv[4]);
//				for (int l=1;l<number_of_layers-1;l++)
//					gaussian_synapses_standard_deviation_E2E_FF[l] = stof(argv[4]);
//				gaussian_synapses_standard_deviation_E2I_L = stof(argv[5]);
				gaussian_synapses_standard_deviation_I2E_L = stof(argv[4]);
				gaussian_synapses_standard_deviation_E2E_L = stof(argv[5]);
				break;
			case OPTIM_DECAY:
				decay_term_tau_g_G2E_FF = stof(argv[4]);
				decay_term_tau_g_E2E_FF = stof(argv[4]);
				decay_term_tau_g_E2E_L = stof(argv[4]);
				decay_term_tau_g_E2E_FB = stof(argv[4]);

				// decay_term_tau_g_E2I_L = stof(argv[5]);
				// decay_term_tau_g_I2E_L = stof(argv[6]);

//				decay_term_tau_C = stof(argv[5]);
//				decay_term_tau_D = stof(argv[6]);
				break;
			case OPTIM_DECAY_AND_BIO_CONST:
				decay_term_tau_g_E2E_FF = stof(argv[4]);
				biological_conductance_scaling_constant_lambda_E2E_FF = stof(argv[5]);
//				decay_term_tau_C = stof(argv[5]);
//				decay_term_tau_D = stof(argv[6]);
				break;
			case OPTIM_STDP:
				learning_rate_rho = stof(argv[4]);
				decay_term_tau_C = stof(argv[5]);
				decay_term_tau_D = stof(argv[5]);
				break;
			case OPTIM_STDP_TAU:
				decay_term_tau_C = stof(argv[4]);
				decay_term_tau_D = stof(argv[5]);
				break;
			case OPTIM_FANINRAD_AND_SYNDECAY:
				gaussian_synapses_standard_deviation_G2E_FF = stof(argv[4]);
				for (int l=1;l<number_of_layers-1;l++)
					gaussian_synapses_standard_deviation_E2E_FF[l] = stof(argv[5]);
				gaussian_synapses_standard_deviation_E2I_L = stof(argv[6]);
				gaussian_synapses_standard_deviation_I2E_L = stof(argv[6]);
				gaussian_synapses_standard_deviation_E2E_L = stof(argv[6]);
				decay_term_tau_C = stof(argv[7]);
				decay_term_tau_D = stof(argv[7]);
				break;
			case OPTIM_FF:
//				gaussian_synapses_standard_deviation_G2E_FF *= stof(argv[4]);
				// for (int l=1;l<number_of_layers-1;l++)
				// 	gaussian_synapses_standard_deviation_E2E_FF[l] *= stof(argv[4]);
				biological_conductance_scaling_constant_lambda_E2E_FF = stof(argv[4]);
				decay_term_tau_g_E2E_FF = stof(argv[5]);
				break;
			case OPTIM_E2E:
				decay_term_tau_g_E2E_FF = stof(argv[4]);
				decay_term_tau_g_E2E_L = stof(argv[4]);
				decay_term_tau_g_E2E_FB = stof(argv[4]);

				biological_conductance_scaling_constant_lambda_E2E_FF = stof(argv[5]);
				biological_conductance_scaling_constant_lambda_E2E_L = stof(argv[5]);
				biological_conductance_scaling_constant_lambda_E2E_FB =  stof(argv[5]);
				break;

			case OPTIM_LAT:
				biological_conductance_scaling_constant_lambda_E2I_L = stof(argv[4]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				biological_conductance_scaling_constant_lambda_I2E_L= stof(argv[5]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
				gaussian_synapses_standard_deviation_E2I_L = stof(argv[6]);
				gaussian_synapses_standard_deviation_I2E_L = stof(argv[7]);
				decay_term_tau_g_E2I_L = stof(argv[8]);
				decay_term_tau_g_I2E_L = stof(argv[9]);

				break;
			case OPTIM_E2E_LAT:
				gaussian_synapses_standard_deviation_E2E_L = stof(argv[4]);
				biological_conductance_scaling_constant_lambda_E2E_L = stof(argv[5]);
				break;
			case OPTIM_E2E_FB:
				gaussian_synapses_standard_deviation_E2E_FB = stof(argv[4]);
				break;
		}


		switch(objective_function){
			case OBJFUNC_AVGFR:
			case OBJFUNC_MAXFR:
				simulate_network_to_train_network = false;
				simulate_network_to_test_trained = false;
				break;

			case OBJFUNC_MAXINFO:
			case OBJFUNC_AVGINFO:
			case OBJFUNC_AVGFR_AND_MAXINFO:
			case OBJFUNC_AVGFR_AND_AVGINFO:
				simulate_network_to_train_network = true;
				simulate_network_to_test_trained = true;
				break;

			case OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY:
			case OBJFUNC_MAXFR_AND_MAXINFO_TRAINEDONLY:
			case OBJFUNC_AVGINFO_TRAINEDONLY:
			case OBJFUNC_MAXINFO_TRAINEDONLY:
			case OBJFUNC_AVGINFO_INPUT_TRAINEDONLY:
				simulate_network_to_test_untrained = false;
				simulate_network_to_train_network = true;
				simulate_network_to_test_trained = true;

		}
	}

        // Create the SpikingModel
        SpikingModel model;
        model.SetTimestep(timestep);

	LIFSpikingNeurons lif_spiking_neurons;
	ImagePoissonInputSpikingNeurons input_neurons;
	ConductanceSpikingSynapses conductance_spiking_synapses;
	EvansSTDP evans_stdp;

        model.spiking_neurons = &lif_spiking_neurons;
        model.input_spiking_neurons = &input_neurons;
        model.spiking_synapses = &conductance_spiking_synapses;
        model.stdp_rule = &evans_stdp;

	/////////// STDP SETUP ///////////
	evans_stdp_parameters_struct STDP_PARAMS;
	STDP_PARAMS.decay_term_tau_C = decay_term_tau_C;
	STDP_PARAMS.decay_term_tau_D = decay_term_tau_D;
	STDP_PARAMS.learning_rate_rho = learning_rate_rho;
	evans_stdp.Set_STDP_Parameters(&conductance_spiking_synapses, &lif_spiking_neurons, &input_neurons, &STDP_PARAMS);

        model.init_backend();

	conductance_spiking_synapses.print_synapse_group_details = false;

	/////////// ADD INPUT NEURONS ///////////
	TimerWithMessages adding_input_neurons_timer("Adding Input Neurons...\n");

	if (is_optimisation)
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("../../MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);
	else
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);

	image_poisson_input_spiking_neuron_parameters_struct image_poisson_input_spiking_group_params;
	image_poisson_input_spiking_group_params.rate = 30.0f;
	input_neurons.AddGroupForEachGaborType(&image_poisson_input_spiking_group_params);

	adding_input_neurons_timer.stop_timer_and_log_time_and_message("Input Neurons Added.", true);


	/////////// ADD NEURONS ///////////
	TimerWithMessages adding_neurons_timer("Adding Neurons...\n");

	lif_spiking_neuron_parameters_struct EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.group_shape[0] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.group_shape[1] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.resting_potential_v0 = -0.074f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.somatic_capcitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS.absolute_refractory_period = absolute_refractory_period;


	lif_spiking_neuron_parameters_struct INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.group_shape[0] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.group_shape[1] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.resting_potential_v0 = -0.082f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.somatic_capcitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS.absolute_refractory_period = absolute_refractory_period;

	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;
	for (int l=0;l<number_of_layers;l++){
		EXCITATORY_NEURONS.push_back(model.AddNeuronGroup(&EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
		INHIBITORY_NEURONS.push_back(model.AddNeuronGroup(&INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
		cout<<"Neuron Group "<<EXCITATORY_NEURONS[l]<<": Excitatory layer "<<l<<endl;
		cout<<"Neuron Group "<<INHIBITORY_NEURONS[l]<<": Inhibitory layer "<<l<<endl;
	}


	adding_neurons_timer.stop_timer_and_log_time_and_message("Neurons Added.", true);


	/////////// ADD SYNAPSES ///////////
	TimerWithMessages adding_synapses_timer("Adding Synapses...\n");


	conductance_spiking_synapse_parameters_struct G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = 1;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = false;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = 0.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = weight_range_bottom;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = weight_range_top;


	conductance_spiking_synapse_parameters_struct E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = E2E_FF_minDelay;//5.0*timestep;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = E2E_FF_maxDelay;//3.0f*pow(10, -3);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_FF;
	if (fanInCount_E2E_FF%max_number_of_connections_per_pair!=0)
	{
		printf("number of total syn connections has to be a multiple of max_number_of_connections_per_pair");
		return 0;
	}
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = true;
//	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_FF[0];
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = 0.0;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_E2E_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = weight_range_bottom;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = weight_range_top;


	conductance_spiking_synapse_parameters_struct E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	if(E2E_FB_ON){
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = E2E_FB_minDelay;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = E2E_FB_maxDelay;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = 1;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = true;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = 0.0;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = weight_range_bottom;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = weight_range_top;
	}


	conductance_spiking_synapse_parameters_struct E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = E2I_L_minDelay; //5.0*timestep;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = E2I_L_maxDelay; //3.0f*pow(10, -3);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = 1;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = false;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = 0.0;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = 0.5;//weight_range_bottom;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = 0.5;//weight_range_top;

	conductance_spiking_synapse_parameters_struct I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = I2E_L_minDelay;//5.0*timestep;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = I2E_L_maxDelay;//3.0f*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = 1;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = false;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = -70.0*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = 0.5;//weight_range_bottom;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = 0.5;//weight_range_top;

	conductance_spiking_synapse_parameters_struct E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS;
	if(E2E_L_ON){
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[0] = E2E_L_minDelay;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.delay_range[1] = E2E_L_maxDelay;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.max_number_of_connections_per_pair = 1;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.stdp_on = E2E_L_STDP_ON;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.reversal_potential_Vhat = 0.0;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.decay_term_tau_g = decay_term_tau_g_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_bottom = weight_range_bottom;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.weight_range_top = weight_range_top;
	}



	for (int l=0; l<number_of_layers; l++){
		if(l==0)
                  model.AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS[l], &G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		else{
			E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS.gaussian_synapses_standard_deviation = gaussian_synapses_standard_deviation_E2E_FF[l-1];
			model.AddSynapseGroup(EXCITATORY_NEURONS[l-1], EXCITATORY_NEURONS[l], &E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
			if(E2E_FB_ON)
				model.AddSynapseGroup(EXCITATORY_NEURONS[l], EXCITATORY_NEURONS[l-1], &E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		}
		model.AddSynapseGroup(EXCITATORY_NEURONS[l], INHIBITORY_NEURONS[l], &E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		model.AddSynapseGroup(INHIBITORY_NEURONS[l], EXCITATORY_NEURONS[l], &I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
		if(E2E_L_ON)
			model.AddSynapseGroup(EXCITATORY_NEURONS[l], EXCITATORY_NEURONS[l], &E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);
	}
	
	adding_synapses_timer.stop_timer_and_log_time_and_message("Synapses Added.", true);


	/////////// SETUP NETWORK ///////////
        model.finalise_model();


	// Create an instance of the Simulator and set the timestep
        Simulator_Options simulator_options;
	Simulator simulator(&model, &simulator_options);
	if (!is_optimisation){ 	// copy cpp file to save parameters for future references
		simulator.CreateDirectoryForSimulationDataFiles(experimentName);
		string source = "Experiments/ConductanceExperiment1.cpp";
		string destination = "output/"+experimentName+"/ConductanceExperiment1.cpp";
		ifstream srce(source.c_str(), ios::binary ) ;
		ofstream dest(destination.c_str(), ios::binary ) ;
		dest << srce.rdbuf() ;
	}


	/////////// SETUP RECORDING ELECTRODES ///////////
	// simulator.setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator.setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);

	float single_score_to_write_to_file_for_dakota_optimisation_excit = 0.0;
	float single_score_to_write_to_file_for_dakota_optimisation_inhib = 0.0;


	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	float presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_test;
	bool record_spikes = record_spikes_test;
	bool save_recorded_spikes_and_states_to_file = save_recorded_spikes_and_states_to_file_test;

	if (is_optimisation)
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("../../MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
	else
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 100.0f);
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);


	SpikeAnalyser spike_analyser_for_untrained_network(model.spiking_neurons, model.input_spiking_neurons, simulator.count_neuron_spikes_recording_electrodes);
	spike_analyser_for_untrained_network.optimal_average_firing_rate = optimal_average_firing_rate;
	spike_analyser_for_untrained_network.optimal_max_firing_rate = optimal_max_firing_rate;
	if (simulate_network_to_test_untrained) {
                // TODO!!!
                simulator.RunSimulation();
		// simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser_for_untrained_network,human_readable_storage,isTrained);
		
		spike_analyser_for_untrained_network.calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		for(int l=0;l<number_of_layers;l++)
			spike_analyser_for_untrained_network.calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS[l], number_of_bins, useThresholdForMaxFR,max_firing_rate);

		isTrained = true;
	}


	/////////// SIMULATE NETWORK TRAINING ///////////

	if (is_optimisation)
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs_train/", 100.0f);
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("../../MatlabGaborFilter/"+inputs_for_train_name+"/").c_str(), 100.0f);
	else
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs_train/", 100.0f);
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("MatlabGaborFilter/"+inputs_for_train_name+"/").c_str(), 100.0f);

	if (simulate_network_to_train_network) {
		presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_train;
		int stimulus_presentation_order_seed = 1;
		int number_of_epochs = number_of_epochs_train;
		Stimuli_Presentation_Struct stimuli_presentation_params;
		stimuli_presentation_params.presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
		stimuli_presentation_params.object_order = OBJECT_ORDER_ORIGINAL;//OBJECT_ORDER_RANDOM;
		stimuli_presentation_params.transform_order = TRANSFORM_ORDER_RANDOM;

                // TODO!!!
                simulator.RunSimulation();
		// simulator.RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch, number_of_epochs, stimuli_presentation_params, stimulus_presentation_order_seed);
	}



	/////////// SIMULATE NETWORK TO TEST TRAINED ///////////
	if (is_optimisation)
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("../../MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
	else
		// input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 100.0f);
		input_neurons.set_up_rates("FileList.txt", "FilterParameters.txt", ("MatlabGaborFilter/"+inputs_for_test_name+"/").c_str(), 100.0f);

	SpikeAnalyser spike_analyser_for_trained_network(model.spiking_neurons, model.input_spiking_neurons, simulator.count_neuron_spikes_recording_electrodes);
	spike_analyser_for_trained_network.optimal_average_firing_rate = optimal_average_firing_rate;
	spike_analyser_for_trained_network.optimal_max_firing_rate = optimal_max_firing_rate;

	if (simulate_network_to_test_trained) {
		presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_test;
		record_spikes = record_spikes_test;
		save_recorded_spikes_and_states_to_file = save_recorded_spikes_and_states_to_file_test;

                // TODO!!!
                simulator.RunSimulation();
		// simulator.RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser_for_trained_network,human_readable_storage,isTrained);

		spike_analyser_for_trained_network.calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
		for(int l=0;l<number_of_layers;l++)
			spike_analyser_for_trained_network.calculate_single_cell_information_scores_for_neuron_group(EXCITATORY_NEURONS[l], number_of_bins,useThresholdForMaxFR,max_firing_rate);
	}

	/////////// PLOT INFOANALYSIS RESULTS //////////////////
	// if (simulate_network_to_test_untrained && simulate_network_to_test_trained && plotInfoAnalysis){
	// 	Plotter plotter(experimentName);
	// 	plotter.plot_single_cell_information_analysis(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);

	// }

	/////////// WRITE INFORMATION ////////////////////
//	if (simulate_network_to_test_untrained && simulate_network_to_test_trained && writeInformation){
//	}


	/////////// WRITE NETWORK SCORE TO RESULTS FILE FOR DAKOTA OPTIMISATION ///////////
	if (is_optimisation){
		TimerWithMessages writing_network_score_to_results_file_timer("Writing Network Score to Results File for Dakota Optimisation...\n");
		float scoreMean_excit = 0;
		float scoreMean_inhib = 0;
		float scoreMax_excit = 0;
		float scoreMax_inhib = 0;
		float combined_information_score_training_increase = 0;
		ofstream resultsfile;
		resultsfile.open(argv[1], ios::out | ios::binary);

		switch (objective_function){
			case OBJFUNC_AVGFR:		//output combined powered distance as a objective function of the optimization
                          spike_analyser_for_untrained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);

				for (int l=0;l<number_of_layers;l++){
					scoreMean_excit += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
					scoreMean_inhib += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
				}
				scoreMean_excit/=number_of_layers;
				scoreMean_inhib/=number_of_layers;
				printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
				resultsfile << to_string(scoreMean_excit) <<endl << to_string(scoreMean_inhib) << endl;
				break;
			case OBJFUNC_MAXFR:
				spike_analyser_for_untrained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				for (int l=0;l<number_of_layers;l++){
					scoreMax_excit += spike_analyser_for_untrained_network.combined_powered_distance_from_max_score_for_each_neuron_group[l*2];
					scoreMax_inhib += spike_analyser_for_untrained_network.combined_powered_distance_from_max_score_for_each_neuron_group[l*2 + 1];
				}
				scoreMax_excit/=number_of_layers;
				scoreMax_inhib/=number_of_layers;
				printf("maxFR score ex: %f inhib: %f \n",scoreMax_excit, scoreMax_inhib);
				resultsfile << to_string(scoreMax_excit) <<endl << to_string(scoreMax_inhib) << endl;
				break;
			case OBJFUNC_MAXINFO:
				//float combined_information_score_training_increase = spike_analyser_for_trained_network.maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network.maximum_information_score_count_multiplied_by_sum_of_information_scores;
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group - spike_analyser_for_untrained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group;
				printf("increase of the number of neurons that have maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;
			case OBJFUNC_AVGINFO:
				//float combined_information_score_training_increase = spike_analyser_for_trained_network.maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network.maximum_information_score_count_multiplied_by_sum_of_information_scores;
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;
			case OBJFUNC_AVGFR_AND_MAXINFO:
				spike_analyser_for_untrained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				for (int l=0;l<number_of_layers;l++){
					scoreMean_excit += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
					scoreMean_inhib += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
				}
				scoreMean_excit/=number_of_layers;
				scoreMean_inhib/=number_of_layers;
				printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
				resultsfile << to_string((scoreMean_excit + scoreMean_inhib)/2) <<endl;

				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group - spike_analyser_for_untrained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group;
				printf("combined_information_score_training_increase: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;

			case OBJFUNC_AVGFR_AND_AVGINFO:
				spike_analyser_for_untrained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				for (int l=0;l<number_of_layers;l++){
					scoreMean_excit += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
					scoreMean_inhib += spike_analyser_for_untrained_network.combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
				}
				scoreMean_excit/=number_of_layers;
				scoreMean_inhib/=number_of_layers;
				printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
				resultsfile << to_string((scoreMean_excit + scoreMean_inhib)/2) <<endl;

				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;

			case OBJFUNC_MAXFR_AND_AVGINFO:
				spike_analyser_for_trained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				scoreMean_excit += spike_analyser_for_trained_network.combined_powered_distance_from_max_score_for_each_neuron_group[(number_of_layers-1)*2];
				printf("maxFR score excit in the last layer: %f \n",scoreMean_excit);
				resultsfile << to_string(scoreMean_excit) <<endl;

				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;

			case OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY:
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;

				spike_analyser_for_trained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				scoreMean_excit += spike_analyser_for_trained_network.combined_powered_distance_from_max_score_for_each_neuron_group[(number_of_layers-1)*2];
				printf("maxFR score excit in the last layer: %f \n",scoreMean_excit);
				resultsfile << to_string(scoreMean_excit) <<endl;
				break;


			case OBJFUNC_MAXFR_AND_MAXINFO_TRAINEDONLY:
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group;
				printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;

				spike_analyser_for_trained_network.calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
				scoreMean_excit += spike_analyser_for_trained_network.combined_powered_distance_from_max_score_for_each_neuron_group[(number_of_layers-1)*2];
				printf("maxFR score excit in the last layer: %f \n",scoreMean_excit);
				resultsfile << to_string(scoreMean_excit) <<endl;
				break;			


			case OBJFUNC_MAXINFO_TRAINEDONLY:
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_last_neuron_group;
				printf("Number of neurons that have maximum info: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
				break;

			case OBJFUNC_AVGINFO_TRAINEDONLY:
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("Number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;
			break;

			case OBJFUNC_AVGINFO_INPUT_TRAINEDONLY:
				spike_analyser_for_trained_network.calculate_single_cell_information_scores_for_neuron_group(0, number_of_bins,useThresholdForMaxFR,max_firing_rate);
				combined_information_score_training_increase = spike_analyser_for_trained_network.number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
				printf("Number of cells with maximum info in average (input layer): %f\n", combined_information_score_training_increase);
				resultsfile << to_string(combined_information_score_training_increase)<<endl;

				break;

		}
		resultsfile.close();

		writing_network_score_to_results_file_timer.stop_timer_and_log_time_and_message("Network Score Written to File.", true);
	}


	/////////// END OF EXPERIMENT ///////////
	experiment_timer.stop_timer_and_log_time_and_message("Experiment Completed.", true);

	return 0;
}
//
