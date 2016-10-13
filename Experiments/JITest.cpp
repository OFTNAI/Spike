#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>
#include "../Plotting/Plotter.h"
#include <vector>
#include "../Helpers/CUDAErrorCheckHelpers.h"

#include <iostream>
using namespace std;

#include "../Models/FourLayerVisionSpikingModel.h"
#include "../Experiments/TestNetworkExperiment.h"
#include "../Experiments/TestTrainTestExperimentSet.h"
#include "../Experiments/CollectEventsNetworkExperiment.h"


// make FILE='JITest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// enum OPTIMISATION_VARIABLES { // E2E ????
// 	OPTIMISATION_VARIABLES_BIOLOGICAL_SCALING_CONSTANT_G2E,
// 	OPTIMISATION_VARIABLES_BIOLOGICAL_SCALING_CONSTANT_E2I,
// 	OPTIMISATION_VARIABLES_BIOLOGICAL_SCALING_CONSTANT_I2E,
// 	OPTIMISATION_VARIABLES_BIOLOGICAL_SCALING_CONSTANT_FF,
// 	OPTIMISATION_VARIABLES_FAN_IN_RADIUS,
// 	OPTIMISATION_VARIABLES_DECAY,
// 	OPTIMISATION_VARIABLES_,
// 	OPTIMISATION_VARIABLES_,
// 	OPTIMISATION_VARIABLES_
// };


// enum OPTIMISATION_OBJECTIVE_FUNCTIONS {
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_AVERAGE_FIRING_RATE,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_MAX_FIRING_RATE,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_,
// 	OPTIMISATION_OBJECTIVE_FUNCTIONS_
// };



// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	TimerWithMessages * experiment_timer = new TimerWithMessages();

	// const int OPTIM_BIO_CONST_LAT = 1;
	// const int OPTIM_BIO_CONST_FF = 2;
	// const int OPTIM_BIO_CONST_LAT_FF = 3;
	// const int OPTIM_FANINRAD = 4;
	// const int OPTIM_DECAY = 5;
	// const int OPTIM_STDP = 6;
	// const int OPTIM_STDP_TAU = 7;
	// const int OPTIM_FANINRAD_AND_SYNDECAY = 8;
	// const int OPTIM_FF = 9;
	// const int OBJFUNC_AVGFR = 1;
	// const int OBJFUNC_MAXFR = 2;
	// const int OBJFUNC_MAXINFO = 3;
	// const int OBJFUNC_AVGINFO = 4;
	// const int OBJFUNC_AVGFR_AND_MAXINFO = 5;
	// const int OBJFUNC_AVGFR_AND_AVGINFO = 6;
	// const int OBJFUNC_MAXFR_AND_AVGINFO = 7;
	// const int OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY = 8;

	// Parameters related to Dakota Optimization
	// int optimizationType = OPTIM_STDP_TAU; //OPTIM_BIO_CONST_LAT, OPTIM_BIO_CONST_FF, OPTIM_BIO_CONST_LAT_FF, OPTIM_FANINRAD, OPTIM_DECAY
	// int objective_function = OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY; //OBJFUNC_AVGFR, OBJFUNC_MAXFR, OBJFUNC_INFO, OBJFUNC_AVGFR_AND_INFO
	// float optimal_average_firing_rate = 10.0f;//set if optimizing based on avgfr : Spontaneous rate (spikes/sec) 4.9 +- 7.1 (*1)
	const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	//*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.

	


	// Simulator Parameters
	// float timestep = 0.00002;
	float timestep = 0.00002;
	bool high_fidelity_spike_storage = true;

	bool simulate_network_to_test_untrained = true;
	bool simulate_network_to_train_network = true;
	bool simulate_network_to_test_trained = true;
	

	bool human_readable_storage = false;
	bool plotInfoAnalysis = true;
	bool writeInformation = true;


	

	// Parameters for testing
	// const float presentation_time_per_stimulus_per_epoch_test = 2.0f;
	const float presentation_time_per_stimulus_per_epoch_test = 0.5f;

	bool record_test_spikes = true;
	bool save_recorded_spikes_and_states_to_file_test = true;

	// Parameters for training
	// float presentation_time_per_stimulus_per_epoch_train = 2.0f;//0.5f;
	float presentation_time_per_stimulus_per_epoch_train = 0.5f;//0.5f;

	int number_of_training_epochs = 10;

	// Parameters for Information Analysis
	int number_of_bins = 5;
	bool useThresholdForMaxFR = true;
	float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;


	// init parameters
	// bool is_optimisation = false;
	bool network_is_trained = false;


	// if (argc > 1) {
		// is_optimisation = true;
		// save_recorded_spikes_and_states_to_file_test = false;
		// plotInfoAnalysis = false;

		// switch (optimizationType){
		// 	case OPTIM_BIO_CONST_LAT:
		// 		biological_conductance_scaling_constant_lambda_E2I_L = stof(argv[4]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		// 		biological_conductance_scaling_constant_lambda_I2E_L= stof(argv[5]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		// 		break;
		// 	case OPTIM_BIO_CONST_FF:
		// 		biological_conductance_scaling_constant_lambda_E2E_FF= stof(argv[4]); //E2E_FF
		// 		break;
		// 	case OPTIM_BIO_CONST_LAT_FF:
		// 		biological_conductance_scaling_constant_lambda_E2E_FF= stof(argv[4]); //E2E_FF
		// 		biological_conductance_scaling_constant_lambda_E2I_L = stof(argv[5]); //E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		// 		biological_conductance_scaling_constant_lambda_I2E_L= stof(argv[6]); //I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS
		// 		break;
		// 	case OPTIM_FANINRAD:
		// 		gaussian_synapses_standard_deviation_G2E_FF = stof(argv[4]);
		// 		for (int l=1;l<four_layer_vision_spiking_model->number_of_non_input_layers-1;l++)
		// 			gaussian_synapses_standard_deviation_E2E_FF[l] = stof(argv[5]);
		// 		gaussian_synapses_standard_deviation_E2I_L = stof(argv[6]);
		// 		gaussian_synapses_standard_deviation_I2E_L = stof(argv[6]);
		// 		gaussian_synapses_standard_deviation_E2E_L = stof(argv[6]);
		// 		break;
		// 	case OPTIM_DECAY:
		// 		decay_term_tau_g_I2E_L = stof(argv[4]);
		// 		decay_term_tau_C = stof(argv[5]);
		// 		decay_term_tau_D = stof(argv[6]);
		// 		break;
		// 	case OPTIM_STDP:
		// 		learning_rate_rho = stof(argv[4]);
		// 		decay_term_tau_C = stof(argv[5]);
		// 		decay_term_tau_D = stof(argv[5]);
		// 		break;
		// 	case OPTIM_STDP_TAU:
		// 		decay_term_tau_C = stof(argv[4]);
		// 		decay_term_tau_D = stof(argv[5]);
		// 		break;
		// 	case OPTIM_FANINRAD_AND_SYNDECAY:
		// 		gaussian_synapses_standard_deviation_G2E_FF = stof(argv[4]);
		// 		for (int l=1;l<four_layer_vision_spiking_model->number_of_non_input_layers-1;l++)
		// 			gaussian_synapses_standard_deviation_E2E_FF[l] = stof(argv[5]);
		// 		gaussian_synapses_standard_deviation_E2I_L = stof(argv[6]);
		// 		gaussian_synapses_standard_deviation_I2E_L = stof(argv[6]);
		// 		gaussian_synapses_standard_deviation_E2E_L = stof(argv[6]);
		// 		decay_term_tau_C = stof(argv[7]);
		// 		decay_term_tau_D = stof(argv[7]);
		// 		break;
		// 	case OPTIM_FF:
		// 		for (int l=1;l<four_layer_vision_spiking_model->number_of_non_input_layers-1;l++)
		// 			gaussian_synapses_standard_deviation_E2E_FF[l] = stof(argv[4]);
		// 		biological_conductance_scaling_constant_lambda_E2E_FF = stof(argv[5]);
		// 		break;

		// }


		// switch(objective_function){
		// 	case OBJFUNC_AVGFR:
		// 	case OBJFUNC_MAXFR:
		// 		simulate_network_to_train_network = false;
		// 		simulate_network_to_test_trained = false;
		// 		break;

		// 	case OBJFUNC_MAXINFO:
		// 	case OBJFUNC_AVGINFO:
		// 	case OBJFUNC_AVGFR_AND_MAXINFO:
		// 	case OBJFUNC_AVGFR_AND_AVGINFO:
		// 		simulate_network_to_train_network = true;
		// 		simulate_network_to_test_trained = true;
		// 		break;

		// 	case OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY:
		// 		simulate_network_to_test_untrained = false;
		// 		simulate_network_to_train_network = true;
		// 		simulate_network_to_test_trained = true;

		// }
	// }

	int random_states_threads_per_block_x = 128;
	int random_states_number_of_blocks_x = 64;
	RandomStateManager::instance()->set_up_random_states(random_states_threads_per_block_x, random_states_number_of_blocks_x, 9);


	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(timestep);
	four_layer_vision_spiking_model->finalise_model();
	four_layer_vision_spiking_model->copy_model_to_device(high_fidelity_spike_storage);



	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	TestNetworkExperiment * test_untrained_network_experiment = new TestNetworkExperiment();
	test_untrained_network_experiment->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
	test_untrained_network_experiment->prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);
	test_untrained_network_experiment->run_experiment(presentation_time_per_stimulus_per_epoch_test, record_test_spikes, save_recorded_spikes_and_states_to_file_test, human_readable_storage, network_is_trained);
	test_untrained_network_experiment->calculate_spike_totals_averages_and_information(number_of_bins, useThresholdForMaxFR, max_firing_rate);





//	CollectEventsNetworkExperiment * collect_events_experiment_set = new CollectEventsNetworkExperiment();
//	collect_events_experiment_set->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
//	collect_events_experiment_set->prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);
//	collect_events_experiment_set->prepare_arrays_for_event_collection(test_untrained_network_experiment);
//	collect_events_experiment_set->run_experiment(presentation_time_per_stimulus_per_epoch_test, network_is_trained);

	// PASS LAST RUN SPIKE_TIMES + IDS



	// TestTrainTestExperimentSet * test_train_test_experiment_set = new TestTrainTestExperimentSet();
	// test_train_test_experiment_set->run_experiment_set_for_model(four_layer_vision_spiking_model, presentation_time_per_stimulus_per_epoch_test, record_test_spikes, save_recorded_spikes_and_states_to_file_test, human_readable_storage, high_fidelity_spike_storage, number_of_bins, useThresholdForMaxFR, max_firing_rate, presentation_time_per_stimulus_per_epoch_train, number_of_training_epochs);


	/////////// PLOT INFOANALYSIS RESULTS //////////////////
	// if (simulate_network_to_test_untrained && simulate_network_to_test_trained && plotInfoAnalysis){
	// 	Plotter * plotter = new Plotter();
	// 	plotter->plot_single_cell_information_analysis(spike_analyser_for_untrained_network, spike_analyser_for_trained_network);

	// }

	/////////// WRITE INFORMATION ////////////////////
//	if (simulate_network_to_test_untrained && simulate_network_to_test_trained && writeInformation){
//	}


	/////////// WRITE NETWORK SCORE TO RESULTS FILE FOR DAKOTA OPTIMISATION ///////////
	// if (is_optimisation){
	// 	TimerWithMessages * writing_network_score_to_results_file_timer = new TimerWithMessages("Writing Network Score to Results File for Dakota Optimisation...\n");
	// 	float scoreMean_excit = 0;
	// 	float scoreMean_inhib = 0;
	// 	float scoreMax_excit = 0;
	// 	float scoreMax_inhib = 0;
	// 	float combined_information_score_training_increase = 0;
	// 	ofstream resultsfile;
	// 	resultsfile.open(argv[1], ios::out | ios::binary);

	// 	switch (objective_function){
	// 		case OBJFUNC_AVGFR:		//output combined powered distance as a objective function of the optimization
	// 			spike_analyser_for_untrained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);

	// 			for (int l=0;l<four_layer_vision_spiking_model->number_of_non_input_layers;l++){
	// 				scoreMean_excit += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
	// 				scoreMean_inhib += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
	// 			}
	// 			scoreMean_excit/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			scoreMean_inhib/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
	// 			resultsfile << to_string(scoreMean_excit) <<endl << to_string(scoreMean_inhib) << endl;
	// 			break;
	// 		case OBJFUNC_MAXFR:
	// 			spike_analyser_for_untrained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
	// 			for (int l=0;l<four_layer_vision_spiking_model->number_of_non_input_layers;l++){
	// 				scoreMax_excit += spike_analyser_for_untrained_network->combined_powered_distance_from_max_score_for_each_neuron_group[l*2];
	// 				scoreMax_inhib += spike_analyser_for_untrained_network->combined_powered_distance_from_max_score_for_each_neuron_group[l*2 + 1];
	// 			}
	// 			scoreMax_excit/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			scoreMax_inhib/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			printf("maxFR score ex: %f inhib: %f \n",scoreMax_excit, scoreMax_inhib);
	// 			resultsfile << to_string(scoreMax_excit) <<endl << to_string(scoreMax_inhib) << endl;
	// 			break;
	// 		case OBJFUNC_MAXINFO:
	// 			//float combined_information_score_training_increase = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;
	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_last_neuron_group - spike_analyser_for_untrained_network->number_of_neurons_with_maximum_information_score_in_last_neuron_group;
	// 			printf("increase of the number of neurons that have maximum info in average: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			break;
	// 		case OBJFUNC_AVGINFO:
	// 			//float combined_information_score_training_increase = spike_analyser_for_trained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores - spike_analyser_for_untrained_network->maximum_information_score_count_multiplied_by_sum_of_information_scores;
	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
	// 			printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			break;
	// 		case OBJFUNC_AVGFR_AND_MAXINFO:
	// 			spike_analyser_for_untrained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
	// 			for (int l=0;l<four_layer_vision_spiking_model->number_of_non_input_layers;l++){
	// 				scoreMean_excit += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
	// 				scoreMean_inhib += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
	// 			}
	// 			scoreMean_excit/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			scoreMean_inhib/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
	// 			resultsfile << to_string((scoreMean_excit + scoreMean_inhib)/2) <<endl;

	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_last_neuron_group - spike_analyser_for_untrained_network->number_of_neurons_with_maximum_information_score_in_last_neuron_group;
	// 			printf("combined_information_score_training_increase: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			break;

	// 		case OBJFUNC_AVGFR_AND_AVGINFO:
	// 			spike_analyser_for_untrained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
	// 			for (int l=0;l<four_layer_vision_spiking_model->number_of_non_input_layers;l++){
	// 				scoreMean_excit += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2];
	// 				scoreMean_inhib += spike_analyser_for_untrained_network->combined_powered_distance_from_average_score_for_each_neuron_group[l*2 + 1];
	// 			}
	// 			scoreMean_excit/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			scoreMean_inhib/=four_layer_vision_spiking_model->number_of_non_input_layers;
	// 			printf("avgFR score ex: %f inhib: %f \n",scoreMean_excit, scoreMean_inhib);
	// 			resultsfile << to_string((scoreMean_excit + scoreMean_inhib)/2) <<endl;

	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
	// 			printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			break;

	// 		case OBJFUNC_MAXFR_AND_AVGINFO:
	// 			spike_analyser_for_trained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
	// 			scoreMean_excit += spike_analyser_for_trained_network->combined_powered_distance_from_max_score_for_each_neuron_group[(four_layer_vision_spiking_model->number_of_non_input_layers-1)*2];
	// 			printf("maxFR score excit in the last layer: %f \n",scoreMean_excit);
	// 			resultsfile << to_string(scoreMean_excit) <<endl;

	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group - spike_analyser_for_untrained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
	// 			printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			break;

	// 		case OBJFUNC_MAXFR_AND_AVGINFO_TRAINEDONLY:
	// 			combined_information_score_training_increase = spike_analyser_for_trained_network->number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
	// 			printf("increase in number of cells with maximum info in average: %f\n", combined_information_score_training_increase);
	// 			resultsfile << to_string(combined_information_score_training_increase)<<endl;
	// 			//spike_analyser_for_trained_network->calculate_fitness_score(optimal_average_firing_rate, optimal_max_firing_rate);
	// 			//scoreMean_excit += spike_analyser_for_trained_network->combined_powered_distance_from_max_score_for_each_neuron_group[(four_layer_vision_spiking_model->number_of_non_input_layers-1)*2];
	// 			//printf("maxFR score excit in the last layer: %f \n",scoreMean_excit);
	// 			//resultsfile << to_string(scoreMean_excit) <<endl;
	// 			break;

	// 	}
	// 	resultsfile.close();

	// 	writing_network_score_to_results_file_timer->stop_timer_and_log_time_and_message("Network Score Written to File.", true);
	// }


	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("TestNetworkExperiment Completed.", true);

	return 0;
}
//
