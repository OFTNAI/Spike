#ifndef FourLayerVisionSpikingModel_H
#define FourLayerVisionSpikingModel_H

#include "SpikingModel.hpp"


class FourLayerVisionSpikingModel : public SpikingModel {

public:

	// Constructor/Destructor
	FourLayerVisionSpikingModel();
	~FourLayerVisionSpikingModel();

        const char * inputs_directory;

	// Network Parameters
	bool INHIBITORY_NEURONS_ON = false;

        bool E2E_FF_SYNAPSES_ON = false;
	bool E2E_L_SYNAPSES_ON = false;
	bool E2I_L_SYNAPSES_ON = false;
	bool I2E_L_SYNAPSES_ON = false;
	bool E2E_FB_SYNAPSES_ON = false;

    STDPPlasticity * E2E_FF_STDP_ON = nullptr;
    STDPPlasticity * E2E_L_STDP_ON = nullptr;
    STDPPlasticity * E2E_FB_STDP_ON = nullptr;
	
	int number_of_non_input_layers = 4;
	int number_of_non_input_layers_to_simulate = 1;
	int dim_excit_layer = 64;
	int dim_inhib_layer = 32;


	// Layer-by-layer parameters
	int * LBL_max_number_of_connections_per_pair_E2E_FF = nullptr;
	int * LBL_max_number_of_connections_per_pair_E2I_L = nullptr;
	int * LBL_max_number_of_connections_per_pair_I2E_L = nullptr;
	int * LBL_max_number_of_connections_per_pair_E2E_L = nullptr;
	int * LBL_max_number_of_connections_per_pair_E2E_FB = nullptr;

	int * LBL_fanInCount_E2E_FF = nullptr;
	int * LBL_fanInCount_E2I_L = nullptr;
	int * LBL_fanInCount_I2E_L = nullptr;
	int * LBL_fanInCount_E2E_L = nullptr;
	int * LBL_fanInCount_E2E_FB = nullptr;

	float * LBL_gaussian_synapses_sd_E2E_FF = nullptr;
	float * LBL_gaussian_synapses_sd_E2I_L = nullptr;
	float * LBL_gaussian_synapses_sd_I2E_L = nullptr;
	float * LBL_gaussian_synapses_sd_E2E_L = nullptr;
	float * LBL_gaussian_synapses_sd_E2E_FB = nullptr;

	float * LBL_biological_conductance_scaling_constant_lambda_E2E_FF = nullptr;
	float * LBL_biological_conductance_scaling_constant_lambda_E2I_L = nullptr;
	float * LBL_biological_conductance_scaling_constant_lambda_I2E_L = nullptr;
	float * LBL_biological_conductance_scaling_constant_lambda_E2E_L = nullptr;
	float * LBL_biological_conductance_scaling_constant_lambda_E2E_FB = nullptr;

	float * LBL_decay_term_tau_g_E2E_FF = nullptr;
	float * LBL_decay_term_tau_g_E2I_L = nullptr;
	float * LBL_decay_term_tau_g_I2E_L = nullptr;
	float * LBL_decay_term_tau_g_E2E_L = nullptr;
	float * LBL_decay_term_tau_g_E2E_FB = nullptr;

	// Neuronal Parameters
	float max_FR_of_input_Gabor = 100.0f;
	float absolute_refractory_period = 0.002;


	//Synaptic Parameters
	float learning_rate_rho = 0.1; // 100.0;
	float decay_term_tau_C = 0.3; //(In Ben's model, tau_C/tau_D = 3/5 v 15/25 v 75/125, and the first one produces the best result)
	float decay_term_tau_D = 0.3;


	float E2E_FF_minDelay = 0.0005;
	float E2E_FF_maxDelay = 0.01;
	float E2I_L_minDelay = 0.0005;
	float E2I_L_maxDelay = 0.01;
	float I2E_L_minDelay = 0.0005;
	float I2E_L_maxDelay = 0.01;
	float E2E_FB_minDelay = 0.0005;
	float E2E_FB_maxDelay = 0.01;
	float E2E_L_minDelay = 0.0005;
	float E2E_L_maxDelay = 0.01;



	LIFSpikingNeurons * lif_spiking_neurons = nullptr;
	ImagePoissonInputSpikingNeurons * image_poisson_input_spiking_neurons = nullptr;
	ConductanceSpikingSynapses * conductance_spiking_synapses = nullptr;
	EvansSTDPPlasticity * evans_stdp = nullptr;



	vector<int> EXCITATORY_NEURONS;
	vector<int> INHIBITORY_NEURONS;


        virtual void set_default_parameter_values();
        void finalise_model() override;
        
        virtual void set_LBL_values_for_pointer_from_layer_to_layer(float value, float* pointer, int start_layer, int end_layer);

        // Regularly used 
        void setup_full_standard_model_using_optimal_parameters();

protected:
        void create_parameter_arrays() override;
        virtual void delete_model_components();
};

#endif
