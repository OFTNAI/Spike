#include "Spike/Simulator/Simulator.hpp"
#include "Spike/Models/FourLayerVisionSpikingModel.hpp"

#include "Spike/Helpers/TimerWithMessages.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"
#include "Spike/Helpers/MemoryUsage.hpp"

// Use the following line to compile the binary
// make FILE='CPUTest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	print_line_of_dashes_with_blank_lines_either_side();

	TimerWithMessages * experiment_timer = new TimerWithMessages();

	// Simulator Parameters
	float timestep = 0.5; // 0.00002;
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

        four_layer_vision_spiking_model->number_of_non_input_layers = 1;
        four_layer_vision_spiking_model->INHIBITORY_NEURONS_ON = false;

        four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0] = 0.01;

        //four_layer_vision_spiking_model->finalise_model();
        four_layer_vision_spiking_model->init_backend(high_fidelity_spike_storage);
        std::cout << "Done init_backend\n"
                  << "model->spiking_neurons->backend: "
                  << four_layer_vision_spiking_model->spiking_neurons->backend()
                  << "\n";

        // CREATE SIMULATOR
        Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);

        // RUN SIMULATION
        simulator->RunSimulation();

        delete four_layer_vision_spiking_model;
        delete simulator;

	print_line_of_dashes_with_blank_lines_either_side();

	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	return 0;
}
