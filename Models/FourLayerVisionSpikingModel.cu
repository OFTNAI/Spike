#include "FourLayerVisionSpikingModel.h"


// FourLayerVisionSpikingModel Constructor
FourLayerVisionSpikingModel::FourLayerVisionSpikingModel () { 

	E2E_L_ON = true;
	E2E_FB_ON = false;
	LI_ON = true;
	E2E_L_STDP_ON = false;

	// Network Parameters
	number_of_layers = 4;
	max_number_of_connections_per_pair = 5;
	dim_excit_layer = 64;
	dim_inhib_layer = 32;

	fanInCount_G2E_FF = 30;
	fanInCount_E2E_FF = 100;
	fanInCount_E2I_L = 30;
	fanInCount_I2E_L = 30;
	fanInCount_E2E_L = 10;
	fanInCount_E2E_FB = 10;



	// Synapse Gaussian Standard Deviations
	gaussian_synapses_sd_G2E_FF = 1.0;
	LBL_gaussian_synapses_sd_E2E_FF = (float*)realloc(LBL_gaussian_synapses_sd_E2E_FF, (number_of_layers - 1)*sizeof(float));
	LBL_gaussian_synapses_sd_E2E_FF[0] = 8.0;
	LBL_gaussian_synapses_sd_E2E_FF[1] = 12.0;
	LBL_gaussian_synapses_sd_E2E_FF[2] = 16.0;
	LBL_gaussian_synapses_sd_E2I_L = (float*)realloc(LBL_gaussian_synapses_sd_E2I_L, (number_of_layers - 1)*sizeof(float));
	LBL_gaussian_synapses_sd_E2I_L[0] = 1.0;
	LBL_gaussian_synapses_sd_E2I_L[1] = 1.0;
	LBL_gaussian_synapses_sd_E2I_L[2] = 1.0;
	LBL_gaussian_synapses_sd_I2E_L = (float*)realloc(LBL_gaussian_synapses_sd_I2E_L, (number_of_layers - 1)*sizeof(float));
	LBL_gaussian_synapses_sd_I2E_L[0] = 8.0;
	LBL_gaussian_synapses_sd_I2E_L[1] = 8.0;
	LBL_gaussian_synapses_sd_I2E_L[2] = 8.0;
	LBL_gaussian_synapses_sd_E2E_L = (float*)realloc(LBL_gaussian_synapses_sd_E2E_L, (number_of_layers - 1)*sizeof(float));
	LBL_gaussian_synapses_sd_E2E_L[0] = 4.0;
	LBL_gaussian_synapses_sd_E2E_L[1] = 4.0;
	LBL_gaussian_synapses_sd_E2E_L[2] = 4.0;
	LBL_gaussian_synapses_sd_E2E_FB = (float*)realloc(LBL_gaussian_synapses_sd_E2E_FB, (number_of_layers - 1)*sizeof(float));
	LBL_gaussian_synapses_sd_E2E_FB[0] = 16.0;
	LBL_gaussian_synapses_sd_E2E_FB[1] = 16.0;
	LBL_gaussian_synapses_sd_E2E_FB[2] = 16.0;

	for (int i = 3; i < number_of_layers - 1; i++) {
		LBL_gaussian_synapses_sd_E2E_FF[i] = 16.0;
		LBL_gaussian_synapses_sd_E2I_L[i] = 1.0;
		LBL_gaussian_synapses_sd_I2E_L[i] = 8.0;
		LBL_gaussian_synapses_sd_E2E_L[i] = 4.0;
		LBL_gaussian_synapses_sd_E2E_FB[i] = 16.0;
	} 

	biological_conductance_scaling_constant_lambda_G2E_FF = 0.00002;
	biological_conductance_scaling_constant_lambda_E2E_FF = 0.0001;
	biological_conductance_scaling_constant_lambda_E2I_L = 0.002;
	biological_conductance_scaling_constant_lambda_I2E_L = 0.004;
	biological_conductance_scaling_constant_lambda_E2E_L = 0.0001;
	biological_conductance_scaling_constant_lambda_E2E_FB = 0.00001;

	decay_term_tau_g_G2E_FF = 0.15;
	decay_term_tau_g_E2E_FF = 0.15;
	decay_term_tau_g_E2I_L = 0.002;
	decay_term_tau_g_I2E_L = 0.025; //In Ben's model, 0.005 v 0.025 and latter produced better result
	decay_term_tau_g_E2E_L = 0.15;
	decay_term_tau_g_E2E_FB = 0.15;

	// Neuronal Parameters
	max_FR_of_input_Gabor = 100.0f;
	absolute_refractory_period = 0.002;

	//Synaptic Parameters
	learning_rate_rho = 0.1/timestep;//100.0;// 0.1;
	decay_term_tau_C = 0.3;//(In Ben's model, tau_C/tau_D = 3/5 v 15/25 v 75/125, and the first one produces the best result)
	decay_term_tau_D = 0.3;

	E2E_FF_minDelay = 5.0*timestep;
	E2E_FF_maxDelay = 0.01;//3.0f*pow(10, -3);
	E2I_L_minDelay = 5.0*timestep;
	E2I_L_maxDelay = 0.01;//3.0f*pow(10, -3);
	I2E_L_minDelay = 5.0*timestep;
	I2E_L_maxDelay = 0.01;//3.0f*pow(10, -3);
	E2E_FB_minDelay = 5.0*timestep;
	E2E_FB_maxDelay = 0.01;
	E2E_L_minDelay = 5.0*timestep;
	E2E_L_maxDelay = 0.01;


}


// FourLayerVisionSpikingModel Destructor
FourLayerVisionSpikingModel::~FourLayerVisionSpikingModel () {


}



void FourLayerVisionSpikingModel::finalise_model(bool is_optimisation) {

	lif_spiking_neurons = new LIFSpikingNeurons();
	image_poisson_input_spiking_neurons = new ImagePoissonInputSpikingNeurons();
	conductance_spiking_synapses = new ConductanceSpikingSynapses();
	evans_stdp = new EvansSTDP();

	spiking_neurons = lif_spiking_neurons;
	spiking_synapses = conductance_spiking_synapses;
	input_spiking_neurons = image_poisson_input_spiking_neurons;
	stdp_rule = evans_stdp; 

	/////////// STDP SETUP ///////////
	evans_stdp_parameters_struct * STDP_PARAMS = new evans_stdp_parameters_struct();
	STDP_PARAMS->decay_term_tau_C = decay_term_tau_C;
	STDP_PARAMS->decay_term_tau_D = decay_term_tau_D;
	STDP_PARAMS->learning_rate_rho = learning_rate_rho;
	evans_stdp->Set_STDP_Parameters((SpikingSynapses *) conductance_spiking_synapses, (SpikingNeurons *) lif_spiking_neurons, (SpikingNeurons *) image_poisson_input_spiking_neurons, (stdp_parameters_struct *) STDP_PARAMS);



	conductance_spiking_synapses->print_synapse_group_details = false;


	/////////// ADD INPUT NEURONS ///////////
	TimerWithMessages * adding_image_poisson_input_spiking_neurons_timer = new TimerWithMessages("Adding Input Neurons...\n");

	if (is_optimisation)
		image_poisson_input_spiking_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "../../MatlabGaborFilter/Inputs/", 100.0f);
	else
		image_poisson_input_spiking_neurons->set_up_rates("FileList.txt", "FilterParameters.txt", "MatlabGaborFilter/Inputs/", 100.0f);

	image_poisson_input_spiking_neuron_parameters_struct * image_poisson_input_spiking_group_params = new image_poisson_input_spiking_neuron_parameters_struct();
	image_poisson_input_spiking_group_params->rate = 30.0f; // ??????
	image_poisson_input_spiking_neurons->AddGroupForEachGaborType(image_poisson_input_spiking_group_params);

	adding_image_poisson_input_spiking_neurons_timer->stop_timer_and_log_time_and_message("Input Neurons Added.", true);



	/////////// ADD NEURONS ///////////
	TimerWithMessages * adding_neurons_timer = new TimerWithMessages("Adding Neurons...\n");

	lif_spiking_neuron_parameters_struct * EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = dim_excit_layer;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.074f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 500.0*pow(10, -12);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 25.0*pow(10, -9);
	EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS->absolute_refractory_period = absolute_refractory_period;


	lif_spiking_neuron_parameters_struct * INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS = new lif_spiking_neuron_parameters_struct();
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[0] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->group_shape[1] = dim_inhib_layer;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->resting_potential_v0 = -0.082f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->threshold_for_action_potential_spike = -0.053f;
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_capcitance_Cm = 214.0*pow(10, -12);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->somatic_leakage_conductance_g0 = 18.0*pow(10, -9);
	INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS->absolute_refractory_period = absolute_refractory_period;

	
	for (int l=0;l<number_of_layers;l++){
		EXCITATORY_NEURONS.push_back(AddNeuronGroup(EXCITATORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
		INHIBITORY_NEURONS.push_back(AddNeuronGroup(INHIBITORY_LIF_SPIKING_NEURON_GROUP_PARAMS));
		cout<<"Neuron Group "<<EXCITATORY_NEURONS[l]<<": Excitatory layer "<<l<<endl;
		cout<<"Neuron Group "<<INHIBITORY_NEURONS[l]<<": Inhibitory layer "<<l<<endl;
	}


	adding_neurons_timer->stop_timer_and_log_time_and_message("Neurons Added.", true);


	/////////// ADD SYNAPSES ///////////
	TimerWithMessages * adding_synapses_timer = new TimerWithMessages("Adding Synapses...\n");


	conductance_spiking_synapse_parameters_struct * G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = timestep;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = gaussian_synapses_sd_G2E_FF;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_G2E_FF;


	conductance_spiking_synapse_parameters_struct * E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_FF_minDelay;//5.0*timestep;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_FF_maxDelay;//3.0f*pow(10, -3);
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_FF;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_E2E_FF;


	conductance_spiking_synapse_parameters_struct * E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	if(E2E_FB_ON){
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_FB_minDelay;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_FB_maxDelay;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = max_number_of_connections_per_pair;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_FB;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = true;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
		E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_E2E_FB;
	}


	conductance_spiking_synapse_parameters_struct * E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2I_L_minDelay; //5.0*timestep;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2I_L_maxDelay; //3.0f*pow(10, -3);
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 1;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2I_L;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
	E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_E2I_L;

	conductance_spiking_synapse_parameters_struct * I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = I2E_L_minDelay;//5.0*timestep;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = I2E_L_maxDelay;//3.0f*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 1;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_I2E_L;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = false;
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = -70.0*pow(10, -3);
	I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_I2E_L;

	conductance_spiking_synapse_parameters_struct * E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS = new conductance_spiking_synapse_parameters_struct();
	if(E2E_L_ON){
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[0] = E2E_L_minDelay;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->delay_range[1] = E2E_L_maxDelay;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->max_number_of_connections_per_pair = 1;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_per_postsynaptic_neuron = fanInCount_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->biological_conductance_scaling_constant_lambda = biological_conductance_scaling_constant_lambda_E2E_L;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->stdp_on = E2E_L_STDP_ON;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->reversal_potential_Vhat = 0.0;
		E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->decay_term_tau_g = decay_term_tau_g_E2E_L;
	}



	

	for (int layer_index = 0; layer_index < number_of_layers; layer_index++) {


		if (layer_index == 0) {

			AddSynapseGroupsForNeuronGroupAndEachInputGroup(EXCITATORY_NEURONS[0], G2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		} else {

			E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_FF[layer_index - 1];
			AddSynapseGroup(EXCITATORY_NEURONS[layer_index - 1], EXCITATORY_NEURONS[layer_index], E2E_FF_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}


		if (LI_ON) {

			E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2I_L[layer_index - 1];
			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], INHIBITORY_NEURONS[layer_index], E2I_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

			I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_I2E_L[layer_index - 1];
			AddSynapseGroup(INHIBITORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index], I2E_L_INHIBITORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}
		
		if(E2E_L_ON) {

			E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_L[layer_index - 1];
			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index], E2E_L_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}

		if (E2E_FB_ON) {

			E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS->gaussian_synapses_standard_deviation = LBL_gaussian_synapses_sd_E2E_FB[layer_index - 1];
			AddSynapseGroup(EXCITATORY_NEURONS[layer_index], EXCITATORY_NEURONS[layer_index - 1], E2E_FB_EXCITATORY_CONDUCTANCE_SPIKING_SYNAPSE_PARAMETERS);

		}
		
	}
	
	adding_synapses_timer->stop_timer_and_log_time_and_message("Synapses Added.", true);

}