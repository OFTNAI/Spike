#ifndef ImagePoissonSpikingNeurons_H
#define ImagePoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "PoissonSpikingNeurons.h"

#include <vector>

#include "Utilities.h"

using namespace std;

struct image_poisson_spiking_neuron_parameters_struct : poisson_spiking_neuron_parameters_struct {
	image_poisson_spiking_neuron_parameters_struct(): orientation(0.0f), wavelength(0.0f), phase(0.0f) { poisson_spiking_neuron_parameters_struct(); }

	float orientation;
	float wavelength;
	float phase;
};


class ImagePoissonSpikingNeurons : public PoissonSpikingNeurons {
public:
	// Constructor/Destructor
	ImagePoissonSpikingNeurons(const char * fileList, const char * filterParameters, const char * inputDirectory);
	~ImagePoissonSpikingNeurons();

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();

	void set_images_from_file_list_and_directory(const char * fileList, const char * filterParameters, const char * inputDirectory);
	void loadFileList(const char * fileList, const char * inputDirectory);
	void load_filter_parameters(const char * filterParameters, const char * inputDirectory);
	void loadInput(const char * inputDirectory);
	void copy_buffer_to_device();
	int mapToV1total_number_of_gabor_types(int orientationIndex, int wavelengthIndex, int phaseIndex);

	//JI VARIABLES
	float * input_rates;
	float * d_input_rates;

	int total_number_of_phases;
	int total_number_of_wavelengths;
	int total_number_of_orientations;
	int image_width;

	int total_number_of_rates;
	int total_number_of_rates_per_image;

	u_short total_number_of_gabor_types;
	u_short total_number_of_input_images;
	u_short total_number_of_objects;

	//OLD VARIABLES
	vector<string> inputNames;

	vector<float> * filterPhases;
	vector<int>  * filterWavelengths;
	vector<float> * filterOrientations;

	vector<vector<vector<vector<float> > > > * buffer; // buffer[fileNr(image_number)][total_number_of_gabor_types(gabor_number)][row][col]
	vector<vector<vector<vector<float> > > > * d_buffer;

	

	u_short total_number_of_transformations;
	
	

};

#endif