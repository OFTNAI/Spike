#include <Spike/Simulator/Simulator.h>
#include <Spike/Models/FourLayerVisionSpikingModel.h>

#include <Spike/SpikeAnalyser/SpikeAnalyser.h>
#include <Spike/Helpers/TimerWithMessages.h>
#include <Spike/Helpers/TerminalHelpers.h>
#include <Spike/Helpers/MemoryUsage.h>

// Use the following line to compile the binary
// make FILE='CPUTest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	print_line_of_dashes_with_blank_lines_either_side();

	TimerWithMessages * experiment_timer = new TimerWithMessages();

	// Simulator Parameters
	float timestep = 0.00002;
	bool high_fidelity_spike_storage = true;

	float presentation_time_per_stimulus_per_epoch = 2.0;

	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options = new Simulator_Options();

	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options->run_simulation_general_options->number_of_epochs = 1;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = false;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 1;

	simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;

	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;

        // MODEL
        FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
        four_layer_vision_spiking_model->SetTimestep(timestep);

        float test_optimisation_parameter_value = (optimisation_parameter_max + optimisation_parameter_min) / 2.0;
			
        four_layer_vision_spiking_model->number_of_non_input_layers = 1;
        four_layer_vision_spiking_model->INHIBITORY_NEURONS_ON = false;

        four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0] = 0.01;

        four_layer_vision_spiking_model->finalise_model();
        four_layer_vision_spiking_model->copy_model_to_device(high_fidelity_spike_storage);

        // CREATE SIMULATOR
        Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);

        // RUN SIMULATION
        SpikeAnalyser * spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
        simulator->RunSimulation(spike_analyser);
        spike_analyser->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);

        delete four_layer_vision_spiking_model;
        delete simulator;
        delete spike_analyser;

        printf("final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]: %.12f\n", final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
        printf("iteration_count_for_optimisation_stage: %d\n", iteration_count_for_optimisation_stage);

	print_line_of_dashes_with_blank_lines_either_side();

	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	return 0;
}
