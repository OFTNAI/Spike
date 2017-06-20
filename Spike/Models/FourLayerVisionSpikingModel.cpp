#include "FourLayerVisionSpikingModel.hpp"


// FourLayerVisionSpikingModel Constructor
FourLayerVisionSpikingModel::FourLayerVisionSpikingModel () { 
        create_parameter_arrays();
        set_default_parameter_values();
}

void FourLayerVisionSpikingModel::create_parameter_arrays() {
	// Layer-by-layer parameter arrays
	LBL_max_number_of_connections_per_pair_E2E_FF = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_max_number_of_connections_per_pair_E2I_L = (int*)malloc(number_of_non_input_layers * sizeof(int));	
	LBL_max_number_of_connections_per_pair_I2E_L = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_max_number_of_connections_per_pair_E2E_L = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_max_number_of_connections_per_pair_E2E_FB = (int*)malloc(number_of_non_input_layers * sizeof(int));

	LBL_fanInCount_E2E_FF = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_fanInCount_E2I_L = (int*)malloc(number_of_non_input_layers * sizeof(int));	
	LBL_fanInCount_I2E_L = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_fanInCount_E2E_L = (int*)malloc(number_of_non_input_layers * sizeof(int));
	LBL_fanInCount_E2E_FB = (int*)malloc(number_of_non_input_layers * sizeof(int));

	LBL_gaussian_synapses_sd_E2E_FF = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_gaussian_synapses_sd_E2I_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_gaussian_synapses_sd_I2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_gaussian_synapses_sd_E2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_gaussian_synapses_sd_E2E_FB = (float*)malloc(number_of_non_input_layers * sizeof(float));

	LBL_biological_conductance_scaling_constant_lambda_E2E_FF = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_biological_conductance_scaling_constant_lambda_E2I_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_biological_conductance_scaling_constant_lambda_I2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_biological_conductance_scaling_constant_lambda_E2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_biological_conductance_scaling_constant_lambda_E2E_FB = (float*)malloc(number_of_non_input_layers * sizeof(float));

	LBL_decay_term_tau_g_E2E_FF = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_decay_term_tau_g_E2I_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_decay_term_tau_g_I2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_decay_term_tau_g_E2E_L = (float*)malloc(number_of_non_input_layers * sizeof(float));
	LBL_decay_term_tau_g_E2E_FB = (float*)malloc(number_of_non_input_layers * sizeof(float));

}

void FourLayerVisionSpikingModel::set_default_parameter_values() {

	// Network Parameters
	dim_excit_layer = 64;
	dim_inhib_layer = 32;
	INHIBITORY_NEURONS_ON = false;

	E2E_FF_SYNAPSES_ON = true;
	E2I_L_SYNAPSES_ON = false;
	I2E_L_SYNAPSES_ON = false;
	E2E_L_SYNAPSES_ON = false;
	E2E_FB_SYNAPSES_ON = false;


	//STDP Parameters
	learning_rate_rho = 0.1;//100.0;// 0.1;
	decay_term_tau_C = 0.3;//(In Ben's model, tau_C/tau_D = 3/5 v 15/25 v 75/125, and the first one produces the best result)
	decay_term_tau_D = 0.3;

	E2E_FF_STDP_ON = nullptr;
	E2E_L_STDP_ON = nullptr;
	E2E_FB_STDP_ON = nullptr;


	// Neuronal Parameters
	max_FR_of_input_Gabor = 100.0f;
	absolute_refractory_period = 0.002;


	// Synaptic delay ranges
	E2E_FF_minDelay = 0.0005;
	E2E_FF_maxDelay = 0.01;

	E2I_L_minDelay = 0.0005;
	E2I_L_maxDelay = 0.01;

	I2E_L_minDelay = 0.0005;
	I2E_L_maxDelay = 0.01;

	E2E_FB_minDelay = 0.0005;
	E2E_FB_maxDelay = 0.01;

	E2E_L_minDelay = 0.0005;
	E2E_L_maxDelay = 0.01;

        /*
        float minDelay = 2*timestep;
        float maxDelay = 4*timestep;
        E2E_FF_minDelay = minDelay;
	E2E_FF_maxDelay = maxDelay;

	E2I_L_minDelay = minDelay;
	E2I_L_maxDelay = maxDelay;

	I2E_L_minDelay = minDelay;
	I2E_L_maxDelay = maxDelay;

	E2E_FB_minDelay = minDelay;
	E2E_FB_maxDelay = maxDelay;

	E2E_L_minDelay = minDelay;
	E2E_L_maxDelay = maxDelay;
        */

	for (int layer_index = 0; layer_index  < number_of_non_input_layers; layer_index ++) {

		// MAX NUMBER OF CONNECTIONS PER PAIR
		LBL_max_number_of_connections_per_pair_E2E_FF[layer_index] = 5;
		LBL_max_number_of_connections_per_pair_E2I_L[layer_index] = 5;	
		LBL_max_number_of_connections_per_pair_I2E_L[layer_index] = 1;
		LBL_max_number_of_connections_per_pair_E2E_L[layer_index] = 1;
		LBL_max_number_of_connections_per_pair_E2E_FB[layer_index] = 5;

		// FAN IN COUNTS
		LBL_fanInCount_E2E_FF[layer_index] = (layer_index == 0) ? 30 : 100;
		LBL_fanInCount_E2I_L[layer_index] = 30;
		LBL_fanInCount_I2E_L[layer_index] = 30;
		LBL_fanInCount_E2E_L[layer_index] = 10;
		LBL_fanInCount_E2E_FB[layer_index] = 10;


		// SYNAPSE GAUSSIAN STANDARD DEVIATIONS
		if (layer_index > 3) {
			LBL_gaussian_synapses_sd_E2E_FF[layer_index] = 16.0;
		} else {
			LBL_gaussian_synapses_sd_E2E_FF[0] = 1.0;
			LBL_gaussian_synapses_sd_E2E_FF[1] = 8.0;
			LBL_gaussian_synapses_sd_E2E_FF[2] = 12.0;
			LBL_gaussian_synapses_sd_E2E_FF[3] = 16.0;
		}

		LBL_gaussian_synapses_sd_E2I_L[layer_index] = 1.0;
		LBL_gaussian_synapses_sd_I2E_L[layer_index] = 8.0;
		LBL_gaussian_synapses_sd_E2E_L[layer_index] = 4.0;
		LBL_gaussian_synapses_sd_E2E_FB[layer_index] = 8.0;


		// BIOLOGICAL SCALING CONSTANTS
		LBL_biological_conductance_scaling_constant_lambda_E2E_FF[layer_index] = (layer_index == 0) ? 0.00002 : 0.0001;
		LBL_biological_conductance_scaling_constant_lambda_E2I_L[layer_index] = 0.002;
		LBL_biological_conductance_scaling_constant_lambda_I2E_L[layer_index] = 0.004;
		LBL_biological_conductance_scaling_constant_lambda_E2E_L[layer_index] = 0.0001;
		LBL_biological_conductance_scaling_constant_lambda_E2E_FB[layer_index] = 0.00001;


		// DECAY TERM TAU G
		LBL_decay_term_tau_g_E2E_FF[layer_index] = 0.15;
		LBL_decay_term_tau_g_E2I_L[layer_index] = 0.005;
		LBL_decay_term_tau_g_I2E_L[layer_index] = 0.005; //In Ben's model, 0.005 v 0.025 and latter produced better result
		LBL_decay_term_tau_g_E2E_L[layer_index] = 0.005;
		LBL_decay_term_tau_g_E2E_FB[layer_index] = 0.005;

	}

}


// FourLayerVisionSpikingModel Destructor
FourLayerVisionSpikingModel::~FourLayerVisionSpikingModel () {
  delete_model_components();
}


void FourLayerVisionSpikingModel::delete_model_components() {
	delete lif_spiking_neurons;
	delete image_poisson_input_spiking_neurons;
	delete conductance_spiking_synapses;
	delete evans_stdp;

        EXCITATORY_NEURONS.clear();
        INHIBITORY_NEURONS.clear();
}


void FourLayerVisionSpikingModel::finalise_model() {
        delete_model_components();
        
	lif_spiking_neurons = new LIFSpikingNeurons();
	image_poisson_input_spiking_neurons = new ImagePoissonInputSpikingNeurons();
	conductance_spiking_synapses = new ConductanceSpikingSynapses();

        spiking_neurons = lif_spiking_neurons;
	spiking_synapses = conductance_spiking_synapses;
	input_spiking_neurons = image_poisson_input_spiking_neurons;


        /////////// STDP SETUP ///////////
	evans_stdp_plasticity_parameters_struct * STDP_PARAMS = new evans_stdp_plasticity_parameters_struct();
	STDP_PARAMS->decay_term_tau_C = decay_term_tau_C;
	STDP_PARAMS->decay_term_tau_D = decay_term_tau_D;
	STDP_PARAMS->learning_rate_rho = learning_rate_rho;
	evans_stdp = new EvansSTDPPlasticity((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) image_poisson_input_spiking_neurons, (stdp_plasticity_parameters_struct *) STDP_PARAMS);
	AddPlasticityRule(evans_stdp);


        // NOW INITIALIZE BACKEND WITH NEW COMPONENTS:
        init_backend();

	conductance_spiking_synapses->print_synapse_group_details = false;


	/////////// ADD INPUT NEURONS ///////////
        #ifndef SILENCE_MODEL_SETUP
	TimerWithMessages * adding_image_poisson_input_spiking_neurons_timer = new TimerWithMessages("Adding Input Neurons...\n");
        #endif

        image_poisson_input_spiking_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", inputs_directory, 100.0f);

	image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = new image_poisson_input_spiking_neuron_parameters_struct();
	image_poisson_input_spiking_group_params->rate = 30.0f; // ??????
	image_poisson_input_spiking_neurons->AddGroupForEachGaborType(image_poisson_input_spiking_group_params);

        image_poisson_input_spiking_neurons->copy_rates_to_device();

        #ifndef SILENCE_MODEL_SETUP
	adding_image_poisson_input_spiking_neurons_timer->stop_timer_and_log_time_and_message("Input Neurons Added.", true);
        #endif


	/////////// ADD NEURONS ///////////
        #ifndef SILENCE_MODEL_SETUP
	TimerWithMessages * adding_neurons_timer = new TimerWithMessages("Adding Neurons...\n");
        #endif

	lif_spiking_neuron_parameters_struct * EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.074f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capacitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->absolute_refractory_period = absolute_refractory_period;

	lif_spiking_neuron_parameters_struct * INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.082f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capacitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->absolute_refractory_period = absolute_refractory_period;

	
	for (int layer_index = 0; layer_index < number_of_non_input_layers_to_simulate; layer_index++){

		EXCITATORY_NEURONS.push_back(AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
                #ifndef SILENCE_MODEL_SETUP
		cout << "Neuron Group " << EXCITATORY_NEURONS[layer_index]
                     << ": Excitatory layer " << layer_index << endl;
                #endif

		if (INHIBITORY_NEURONS_ON) {
			INHIBITORY_NEURONS.push_back(AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
                        #ifndef SILENCE_MODEL_SETUP
			cout << "Neuron Group "
                             << INHIBITORY_NEURONS[layer_index]
                             << ": Inhibitory layer " << layer_index << endl;
                        #endif
		}
		
	}

        #ifndef SILENCE_MODEL_SETUP
	adding_neurons_timer->stop_timer_and_log_time_and_message("Neurons Added.", true);
        #endif


	/////////// ADD SYNAPSES ///////////
        #ifndef SILENCE_MODEL_SETUP
	TimerWithMessages * adding_synapses_timer = new TimerWithMessages("Adding Synapses...\n");
        #endif


	conductance_spiking_synapse_parameters_struct * G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_E2E_FF[0];
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_E2E_FF[0];
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0];
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(nullptr);
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_FF[0];
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_E2E_FF[0];


	conductance_spiking_synapse_parameters_struct * E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_FF_minDelay;//5.0*timestep;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_FF_maxDelay;//3.0f*pow(10, -3);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(E2E_FF_STDP_ON);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_FB_minDelay;
	E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_FB_maxDelay;
	E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(E2E_FB_STDP_ON);
	E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2I_L_minDelay; //5.0*timestep;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2I_L_maxDelay; //3.0f*pow(10, -3);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(nullptr);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;

	conductance_spiking_synapse_parameters_struct * I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = I2E_L_minDelay;//5.0*timestep;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = I2E_L_maxDelay;//3.0f*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(nullptr);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);

	conductance_spiking_synapse_parameters_struct * E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_L_minDelay;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_L_maxDelay;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->plasticity_vec.push_back(E2E_L_STDP_ON);
	E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	

	for (int layer_index = 0; layer_index < number_of_non_input_layers_to_simulate; layer_index++) {

#ifdef CRAZY_DEBUG
          std::cout << "layer_index " << layer_index << "\n";
          std::cout << "EXCITATORY_NEURONS[l] = "
                    << EXCITATORY_NEURONS[layer_index]
                    << "\n";
#endif

		if (E2E_FF_SYNAPSES_ON) {
			if (layer_index == 0) {
				AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS[0], G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

			} else {

				E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_E2E_FF[layer_index];
				E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_FF[layer_index];
				E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_E2E_FF[layer_index];
				E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_E2E_FF[layer_index];
				E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_E2E_FF[layer_index];

				AddSynapseGroup(EXCITATORY_NEURONS[layer_index - 1], EXCITATORY_NEURONS[layer_index], E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

			}
		}	

		if (E2I_L_SYNAPSES_ON && INHIBITORY_NEURONS_ON) {

			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_E2I_L[layer_index];
			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2I_L[layer_index];
			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_E2I_L[layer_index];
			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_E2I_L[layer_index];
			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_E2I_L[layer_index];

			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], INHIBITORY_NEURONS[layer_index], E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}

		if(E2E_L_SYNAPSES_ON) {

			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_E2E_L[layer_index];
			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_L[layer_index];
			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_E2E_L[layer_index];
			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_E2E_L[layer_index];
			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_E2E_L[layer_index];

			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index], E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}

		if (I2E_L_SYNAPSES_ON && INHIBITORY_NEURONS_ON) {

			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_I2E_L[layer_index];
			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_I2E_L[layer_index];
			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_I2E_L[layer_index];
			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_I2E_L[layer_index];
			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_I2E_L[layer_index];

			AddSynapseGroup(INHIBITORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index], I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}

		if (E2E_FB_SYNAPSES_ON) {

			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = LBL_fanInCount_E2E_FB[layer_index];
			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_FB[layer_index];
			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = LBL_biological_conductance_scaling_constant_lambda_E2E_FB[layer_index];
			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = LBL_decay_term_tau_g_E2E_FB[layer_index];
			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = LBL_max_number_of_connections_per_pair_E2E_FB[layer_index];

			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index - 1], E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}
		
	}
	
        #ifndef SILENCE_MODEL_SETUP
	adding_synapses_timer->stop_timer_and_log_time_and_message("Synapses Added.", true);
        #endif

        // Update backend after having constructed model:
        prepare_backend();
}


void FourLayerVisionSpikingModel::set_LBL_values_for_pointer_from_layer_to_layer(float value, float* pointer, int start_layer, int end_layer) {

	for (int layer_index = start_layer; layer_index <= end_layer; layer_index++) {
		pointer[layer_index] = value;
	}

}



//Regularly Used Helper
void FourLayerVisionSpikingModel::setup_full_standard_model_using_optimal_parameters() {

        SetTimestep(0.00002);
	number_of_non_input_layers_to_simulate = 1;

	// INHIBITORY_NEURONS_ON = true;
	E2E_FF_SYNAPSES_ON = true;
	// E2I_L_SYNAPSES_ON = true;
	// I2E_L_SYNAPSES_ON = true;
	// E2E_L_SYNAPSES_ON = true;

	LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0] = 0.000292968762;
	LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1] = 0.000030517578;
	LBL_biological_conductance_scaling_constant_lambda_E2E_FF[2] = 0.000036621095;
	LBL_biological_conductance_scaling_constant_lambda_E2E_FF[3] = 0.000061035156;
	set_LBL_values_for_pointer_from_layer_to_layer(0.010937500745, LBL_biological_conductance_scaling_constant_lambda_E2I_L, 0, 3);
	set_LBL_values_for_pointer_from_layer_to_layer(0.050000000745, LBL_biological_conductance_scaling_constant_lambda_I2E_L, 0, 3);
	set_LBL_values_for_pointer_from_layer_to_layer(0.000292968762, LBL_biological_conductance_scaling_constant_lambda_E2E_L, 0, 3);
 
 }
