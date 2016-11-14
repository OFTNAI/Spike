#include "../Optimiser/Optimiser.h"

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/MemoryUsage.h"

#include "cuda_profiler_api.h"


// Use the following line to compile the binary
// make FILE='TestFourLayerVisionSpikingModelFiringRates' EXPERIMENT_DIRECTORY='Experiments'  model -j22


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){


	TimerWithMessages * experiment_timer = new TimerWithMessages("\n");

	const float presentation_time_per_stimulus_per_epoch = 0.2;

	// MODEL
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->setup_full_standard_model_using_optimal_parameters();



	////////////////////////////////
	/// TESTING
	///////////////////////////////

	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options_TEST = new Simulator_Options();

	simulator_options_TEST->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options_TEST->run_simulation_general_options->number_of_epochs = 1;
	simulator_options_TEST->run_simulation_general_options->apply_stdp_to_relevant_synapses = false;
	simulator_options_TEST->run_simulation_general_options->stimulus_presentation_order_seed = 1;

	simulator_options_TEST->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;

	simulator_options_TEST->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	simulator_options_TEST->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options_TEST->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;


	// CREATE SIMULATOR
	Simulator * simulator_TEST = new Simulator(four_layer_vision_spiking_model, simulator_options_TEST);

	// RUN SIMULATION
	SpikeAnalyser * spike_analyser_TEST = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
	simulator_TEST->RunSimulation(spike_analyser_TEST);


	// CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
	spike_analyser_TEST->calculate_various_neuron_spike_totals_and_averages(simulator_options_TEST->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);
	spike_analyser_TEST->calculate_single_cell_information_scores_for_neuron_group(6, 5, false, 0.0);



	////////////////////////////////
	/// TRAINING
	///////////////////////////////


	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options_TRAIN = new Simulator_Options();

	simulator_options_TRAIN->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options_TRAIN->run_simulation_general_options->number_of_epochs = 2;
	simulator_options_TRAIN->run_simulation_general_options->apply_stdp_to_relevant_synapses = true;
	simulator_options_TRAIN->run_simulation_general_options->stimulus_presentation_order_seed = 1;

	simulator_options_TRAIN->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	simulator_options_TRAIN->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options_TRAIN->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;


	// CREATE SIMULATOR
	Simulator * simulator_TRAIN = new Simulator(four_layer_vision_spiking_model, simulator_options_TRAIN);

	// RUN SIMULATION
	SpikeAnalyser * spike_analyser_TRAIN = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
	simulator_TRAIN->RunSimulation(spike_analyser_TRAIN);




	////////////////////////////////
	/// TESTING
	///////////////////////////////

	// CREATE SIMULATOR
	Simulator * simulator_TEST_2 = new Simulator(four_layer_vision_spiking_model, simulator_options_TEST);

	// RUN SIMULATION
	SpikeAnalyser * spike_analyser_TEST_2 = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
	simulator_TEST_2->RunSimulation(spike_analyser_TEST_2);


	// CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
	spike_analyser_TEST_2->calculate_various_neuron_spike_totals_and_averages(simulator_options_TEST->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);
	spike_analyser_TEST_2->calculate_single_cell_information_scores_for_neuron_group(6, 5, false, 0.0);







	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	cudaProfilerStop();

	return 0;
}