#ifndef ImagePoissonInputSpikingNeurons_H
#define ImagePoissonInputSpikingNeurons_H

#define SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "PoissonInputSpikingNeurons.h"

#include <vector>
#include <string>

// using namespace std;

struct image_poisson_input_spiking_neuron_parameters_struct : poisson_input_spiking_neuron_parameters_struct {
	image_poisson_input_spiking_neuron_parameters_struct(): gabor_index(-1) { poisson_input_spiking_neuron_parameters_struct(); }

	int gabor_index;
};


class ImagePoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
public:
	// Constructor/Destructor
	ImagePoissonInputSpikingNeurons();
	~ImagePoissonInputSpikingNeurons();

	virtual int AddGroup(neuron_parameters_struct * group_params);
	void AddGroupForEachGaborType(neuron_parameters_struct * group_params);

	virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);

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

	//OLD VARIABLES
	std::vector<std::string> inputNames;

	std::vector<float> * filterPhases;
	std::vector<int>  * filterWavelengths;
	std::vector<float> * filterOrientations;
	

	
	
};

#endif
