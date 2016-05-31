#ifndef ImagePoissonSpikingNeurons_H
#define ImagePoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "PoissonSpikingNeurons.h"

#include <vector>

using namespace std;

struct image_poisson_spiking_neuron_parameters_struct : poisson_spiking_neuron_parameters_struct {
	image_poisson_spiking_neuron_parameters_struct(): gabor_index(-1) { poisson_spiking_neuron_parameters_struct(); }

	int gabor_index;
};


class ImagePoissonSpikingNeurons : public PoissonSpikingNeurons {
public:
	// Constructor/Destructor
	ImagePoissonSpikingNeurons();
	~ImagePoissonSpikingNeurons();

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	void AddGroupForEachGaborType(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();
	virtual void update_membrane_potentials(float timestep);

	void set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory, float max_rate_scaling_factor);

	void load_image_names_from_file_list(const char * fileList, const char * inputDirectory);
	void load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory);
	void load_rates_from_files(const char * inputDirectory, float max_rate_scaling_factor);
	void copy_rates_to_device();
	int calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex);

	//JI VARIABLES
	float * gabor_input_rates;
	float * d_gabor_input_rates;

	int total_number_of_phases;
	int total_number_of_wavelengths;
	int total_number_of_orientations;
	int image_width;

	int total_number_of_rates;
	int total_number_of_rates_per_image;

	int total_number_of_gabor_types;
	int total_number_of_objects;

	//OLD VARIABLES
	vector<string> inputNames;

	vector<float> * filterPhases;
	vector<int>  * filterWavelengths;
	vector<float> * filterOrientations;
	

	int total_number_of_transformations_per_object;
	
};

__global__ void image_poisson_update_membrane_potentials_kernal(curandState_t* d_states,
							float *d_gabor_input_rates,
							float *d_membrane_potentials_v,
							float timestep,
							float * d_thresholds_for_action_potential_spikes,
							size_t total_number_of_inputs);

#endif