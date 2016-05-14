#include "ImagePoissonSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle

#include <fstream>
#include <sstream>
#include <string>
#include "../Helpers/FstreamWrapper.h"


// ImagePoissonSpikingNeurons Constructor
ImagePoissonSpikingNeurons::ImagePoissonSpikingNeurons() {

	//JI
	total_number_of_input_images = 0;
	total_number_of_transformations_per_object = 0;
	total_number_of_objects = 0;

	total_number_of_phases = 0;
	total_number_of_wavelengths = 0;
	total_number_of_orientations = 0;
	total_number_of_gabor_types = 0;

	image_width = 0;

	total_number_of_rates = 0;
	total_number_of_rates_per_image = 0;

	filterPhases = new vector<float>();
	filterWavelengths = new vector<int>();
	filterOrientations = new vector<float>();

}


// ImagePoissonSpikingNeurons Destructor
ImagePoissonSpikingNeurons::~ImagePoissonSpikingNeurons() {

}


int ImagePoissonSpikingNeurons::AddGroup(neuron_parameters_struct * group_params, int group_shape[2]){

	int new_group_id = PoissonSpikingNeurons::AddGroup(group_params, group_shape);

	image_poisson_spiking_neuron_parameters_struct * image_poisson_spiking_group_params = (image_poisson_spiking_neuron_parameters_struct*)group_params;

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {

	}

	return new_group_id;

}


void ImagePoissonSpikingNeurons::AddGroupForEachGaborType(neuron_parameters_struct * group_params) {

	int group_shape[] = {image_width, image_width};

	image_poisson_spiking_neuron_parameters_struct * image_poisson_spiking_group_params = (image_poisson_spiking_neuron_parameters_struct*)group_params;

	for (int gabor_index = 0; gabor_index < total_number_of_gabor_types; gabor_index++) {
		image_poisson_spiking_group_params->gabor_index = gabor_index;
		int new_group_id = this->AddGroup(image_poisson_spiking_group_params, group_shape);
	}

}


void ImagePoissonSpikingNeurons::allocate_device_pointers() {

	PoissonSpikingNeurons::allocate_device_pointers();

}


void ImagePoissonSpikingNeurons::reset_neurons() {

	PoissonSpikingNeurons::reset_neurons();

}

void ImagePoissonSpikingNeurons::set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory) {
	printf("--- Setting up Input Neuron Rates from Gabor files...\n");

	load_image_names_from_file_list(fileList, inputDirectory);
	load_gabor_filter_parameters(filterParameters, inputDirectory);
	load_rates_from_files(inputDirectory);
	copy_rates_to_device();

	// printf("\n");
}


void ImagePoissonSpikingNeurons::load_image_names_from_file_list(const char * fileList, const char * inputDirectory) {
    
	// Open file list
	stringstream path;
	path << inputDirectory << '/' << fileList;
	string path_string = path.str();
	
	ifstream fileListStream;
	fileListStream.open(path_string);

	if(fileListStream.fail()) {
		stringstream s;
		s << "Unable to open " << path_string << " for input." << endl;
		cerr << s.str();
		exit(EXIT_FAILURE);
	}
	
	string dirNameBase;						// The "shapeS1T2" part of "shapeS1T2.png"
	int filesLoaded = 0;
	int lastNrOfTransformsFound = 0; // For validation of file list
	
	// cout << "Reading file list:" << endl;
	
	while(getline(fileListStream, dirNameBase)) { 	// Read line from file list

		// printf("total_number_of_transformations_per_object: %d\n", total_number_of_transformations_per_object);
		
		if(dirNameBase.compare("") == 0) {
			continue; // Last line may just be empty bcs of matlab script, should be break; really, but what the hell		
		} else if(dirNameBase.compare("*") == 0) {	
			if(lastNrOfTransformsFound != 0 && lastNrOfTransformsFound != total_number_of_transformations_per_object) {
				cerr << "Number of transforms varied in file list" << endl;
				exit(EXIT_FAILURE);
			}
				
			total_number_of_objects++;
			lastNrOfTransformsFound = total_number_of_transformations_per_object;
			total_number_of_transformations_per_object = 0;
			
			continue;
		} else {
			filesLoaded++;
			total_number_of_transformations_per_object++;
		}
		
		// cout << "#" << filesLoaded << " Loading: " << dirNameBase << endl;
		
		inputNames.push_back(dirNameBase);
	}
	
	total_number_of_transformations_per_object = lastNrOfTransformsFound;
	
	cout << "--- --- Objects: " << total_number_of_objects << ", Transforms per Object: " << total_number_of_transformations_per_object << endl;
	
	total_number_of_input_images = total_number_of_objects * total_number_of_transformations_per_object;
}


void ImagePoissonSpikingNeurons::load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory) {


	// cout << "Reading filter parameters:" << endl;

	// Open filterParameters
	stringstream path;
	path << inputDirectory << '/' << filterParameters;
	string path_string = path.str();
	
	ifstream filterParametersStream;
	filterParametersStream.open(path_string);

	if(filterParametersStream.fail()) {
		stringstream s;
		s << "Unable to open " << path_string << " for input." << endl;
		cerr << s.str();
		exit(EXIT_FAILURE);
	}

	string dirNameBase;

	cout << "--- --- Gabor Parameters:" << endl;

	int line_index = 0;
	while(getline(filterParametersStream, dirNameBase)) {

		cout << "--- --- --- " << dirNameBase << endl;

		stringstream lineStream(dirNameBase);

		int num;
		while (lineStream.str().size() != 0) {

			if ((lineStream.peek() == ',') || (lineStream.peek() == '[') || (lineStream.peek() == ' ')) {
				lineStream.ignore();
			} else if (lineStream.peek() == ']') {
				break;
			} else {

				lineStream >> num;

				switch (line_index) {
					case 0:
						filterPhases->push_back((float)num);
						break;

					case 1:
						filterWavelengths->push_back(num);
						break;

					case 2:
						filterOrientations->push_back((float)num);
						break;	
					case 3:
						image_width = num;
						break; 
				}

			}	
		} 

		line_index++;

	}

	total_number_of_phases = filterPhases->size();
	total_number_of_wavelengths = filterWavelengths->size();
	total_number_of_orientations = filterOrientations->size();
	total_number_of_gabor_types = total_number_of_phases*total_number_of_wavelengths*total_number_of_orientations;

	total_number_of_rates_per_image = total_number_of_gabor_types * image_width * image_width;
	total_number_of_rates = total_number_of_input_images * total_number_of_rates_per_image;

	// printf("\ntotal_number_of_rates: %d\n", total_number_of_rates);
}


void ImagePoissonSpikingNeurons::load_rates_from_files(const char * inputDirectory) {


	gabor_input_rates = (float *)malloc(total_number_of_rates*sizeof(float));
	int zero_count = 0;

	for(int image_index = 0; image_index < total_number_of_input_images; image_index++) {

		int image_starting_index = image_index * total_number_of_rates_per_image;
		
		// cout << "Loading Rates for Image #" << image_index << endl;
		
		for(int orientation_index = 0; orientation_index < total_number_of_orientations; orientation_index++) {

			for(int wavelength_index = 0; wavelength_index < total_number_of_wavelengths; wavelength_index++) {

				for(int phase_index = 0; phase_index < total_number_of_phases; phase_index++) {

					int gabor_index = calculate_gabor_index(orientation_index,wavelength_index,phase_index);
					int start_index_for_current_gabor_image = image_starting_index + gabor_index * image_width * image_width;

					// printf("ORIENTATION: %d\n", orientation_index);
					// printf("WAVELENGTH: %d\n", wavelength_index);
					// printf("PHASE: %d\n\n", phase_index);
					// printf("GABOR_INDEX: %d\n", gabor_index);
					
					// Read input to network
					ostringstream dirStream;

					dirStream << inputDirectory << "Filtered/" << inputNames[image_index] << ".flt" << "/"
					<< inputNames[image_index] << '.' << filterWavelengths->at(wavelength_index) << '.' 
					<< filterOrientations->at(orientation_index) << '.' << filterPhases->at(phase_index) << ".gbo";
					
					string t = dirStream.str();
					
					// Open&Read gabor filter file
					fstreamWrapper gaborStream;
					
					try {
						
						gaborStream.open(t.c_str(), std::ios_base::in | std::ios_base::binary);

						for(int image_x = 0; image_x < image_width; image_x++)
							for(int image_y = 0; image_y < image_width; image_y++) {
								
								float rate;
								gaborStream >> rate;
								// printf("rate: %f\n", rate);
								if (rate < 0.000001) zero_count++;
								if(rate < 0) {
									cerr << "Negative firing loaded from filter!!!" << endl;
									exit(EXIT_FAILURE);
								}

								int element_index = start_index_for_current_gabor_image + image_x + image_y * image_width;
								
								gabor_input_rates[element_index] = rate*100.0f;
							}
						
					} catch (fstream::failure e) {
						stringstream s;
						s << "Unable to open/read from " << t << " for gabor input: " << e.what();
						cerr << s.str();
						exit(EXIT_FAILURE);
					}
				}
			}
		}
	}

	printf("--- --- Proportion of input rates 0.0: %f\n", (float)zero_count/(float)total_number_of_rates);
}

void ImagePoissonSpikingNeurons::copy_rates_to_device() {
	CudaSafeCall(cudaMalloc((void **)&d_gabor_input_rates, sizeof(float)*total_number_of_rates));
	CudaSafeCall(cudaMemcpy(d_gabor_input_rates, gabor_input_rates, sizeof(float)*total_number_of_rates, cudaMemcpyHostToDevice));
}


int ImagePoissonSpikingNeurons::calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex) {
	
	return orientationIndex * (total_number_of_wavelengths * total_number_of_phases) + wavelengthIndex * total_number_of_phases + phaseIndex;
}


__global__ void image_poisson_update_membrane_potentials_kernal(curandState_t* d_states,
							float *d_gabor_input_rates,
							float *d_membrane_potentials_v,
							float timestep,
							size_t total_number_of_inputs);


void ImagePoissonSpikingNeurons::update_membrane_potentials(float timestep) {

	image_poisson_update_membrane_potentials_kernal<<<random_state_manager->block_dimensions, random_state_manager->threads_per_block>>>(random_state_manager->d_states,
														d_gabor_input_rates,
														d_membrane_potentials_v,
														timestep,
														total_number_of_neurons);

	CudaCheckError();
}


__global__ void image_poisson_update_membrane_potentials_kernal(curandState_t* d_states,
							float *d_gabor_input_rates,
							float *d_membrane_potentials_v,
							float timestep,
							size_t total_number_of_inputs){

	 
	int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = t_idx;
	while (idx < total_number_of_inputs){

		float rate = d_gabor_input_rates[idx];

		if (rate > 0.000001) {
			// rate = rate * 10.0f;

			// Creates random float between 0 and 1 from uniform distribution
			// d_states effectively provides a different seed for each thread
			// curand_uniform produces different float every time you call it
			float random_float = curand_uniform(&d_states[t_idx]);

			// if (idx == 400) printf("rate: %f, random_float: %f\n", rate, random_float);

			// if the randomnumber is less than the rate
			if (random_float < (rate * timestep)){

				// Puts membrane potential above default spiking threshold
				d_membrane_potentials_v[idx] = 35.0f;

			} 

		}

		idx += blockDim.x * gridDim.x;

	}
	__syncthreads();
}

