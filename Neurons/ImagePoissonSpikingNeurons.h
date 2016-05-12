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
	u_short mapToV1Depth(u_short orientationIndex, u_short wavelengthIndex, u_short phaseIndex);

	vector<string> inputNames;

	vector<float> * filterPhases;
	vector<int>  * filterWavelengths;
	vector<float> * filterOrientations;

	vector<vector<vector<vector<float> > > > buffer; // buffer[fileNr][depth][row][col]

	u_short dimension, depth;
	u_short nrOfTransformations;
	u_short nrOfObjects;
	

};

#endif