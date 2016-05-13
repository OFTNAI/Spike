#ifndef ImagePoissonSpikingNeurons_H
#define ImagePoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "PoissonSpikingNeurons.h"

#include <vector>

// #include "Utilities.h"

using namespace std;

struct image_poisson_spiking_neuron_parameters_struct : poisson_spiking_neuron_parameters_struct {
	image_poisson_spiking_neuron_parameters_struct(): input_image_index(-1) { poisson_spiking_neuron_parameters_struct(); }

	int input_image_index;
};


class ImagePoissonSpikingNeurons : public PoissonSpikingNeurons {
public:
	// Constructor/Destructor
	ImagePoissonSpikingNeurons(const char * fileList, const char * filterParameters, const char * inputDirectory);
	~ImagePoissonSpikingNeurons();

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	void AddGroupForEachInputImage(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();

	void set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory);

	void load_image_names_from_file_list(const char * fileList, const char * inputDirectory);
	void load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory);
	void load_rates_from_files(const char * inputDirectory);
	void copy_rates_to_device();
	int calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex);

	//JI VARIABLES
	float * input_rates;
	float * d_input_rates;

	int total_number_of_phases;
	int total_number_of_wavelengths;
	int total_number_of_orientations;
	int image_width;

	int total_number_of_rates;
	int total_number_of_rates_per_image;

	int total_number_of_gabor_types;
	int total_number_of_input_images;
	int total_number_of_objects;

	//OLD VARIABLES
	vector<string> inputNames;

	vector<float> * filterPhases;
	vector<int>  * filterWavelengths;
	vector<float> * filterOrientations;

	vector<vector<vector<vector<float> > > > * buffer; // buffer[fileNr(image_number)][total_number_of_gabor_types(gabor_number)][row][col]
	vector<vector<vector<vector<float> > > > * d_buffer;

	

	int total_number_of_transformations_per_object;
	
	

};

#endif