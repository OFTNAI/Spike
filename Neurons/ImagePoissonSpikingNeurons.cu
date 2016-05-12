#include "ImagePoissonSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle

#include <fstream>
#include <sstream>
#include <string>
#include "FstreamWrapper.h"


// ImagePoissonSpikingNeurons Constructor
ImagePoissonSpikingNeurons::ImagePoissonSpikingNeurons(const char * fileList, const char * filterParameters, const char * inputDirectory) {

	filterPhases = new vector<float>();
	filterWavelengths = new vector<int>();
	filterOrientations = new vector<float>();
	buffer = new vector<vector<vector<vector<float> > > >();
	nrOfTransformations = 0;
	nrOfObjects = 0;

	set_images_from_file_list_and_directory(fileList, filterParameters, inputDirectory);

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

void ImagePoissonSpikingNeurons::allocate_device_pointers() {

	PoissonSpikingNeurons::allocate_device_pointers();

}


void ImagePoissonSpikingNeurons::reset_neurons() {

	PoissonSpikingNeurons::reset_neurons();

}

void ImagePoissonSpikingNeurons::set_images_from_file_list_and_directory(const char * fileList, const char * filterParameters, const char * inputDirectory) {
	// printf("Set images from file list: %s and directory: %s\n", fileList, inputDirectory);

	loadFileList(fileList, inputDirectory);
	load_filter_parameters(filterParameters, inputDirectory);
	loadInput(inputDirectory);
	copy_buffer_to_device();

	printf("\n");
}


void ImagePoissonSpikingNeurons::loadFileList(const char * fileList, const char * inputDirectory) {
    
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
	u_short filesLoaded = 0;
	u_short lastNrOfTransformsFound = 0; // For validation of file list
	
	cout << "Reading file list:" << endl;


	//JI TEMP
	int nrOfFiles = 0;

	
	while(getline(fileListStream, dirNameBase)) { 	// Read line from file list
		
		if(dirNameBase.compare("") == 0) {
			continue; // Last line may just be empty bcs of matlab script, should be break; really, but what the hell		
		} else if(dirNameBase.compare("*") == 0) {	
			if(lastNrOfTransformsFound != 0 && lastNrOfTransformsFound != nrOfTransformations) {
				cerr << "Number of transforms varied in file list" << endl;
				exit(EXIT_FAILURE);
			}
				
			nrOfObjects++;
			lastNrOfTransformsFound = nrOfTransformations;
			nrOfTransformations = 0;
			
			continue;
		} else {
			filesLoaded++;
			nrOfTransformations++;
		}
		
		cout << "#" << filesLoaded << " Loading: " << dirNameBase << endl;
		
		inputNames.push_back(dirNameBase);
	}
	
	//JI
	nrOfTransformations = lastNrOfTransformsFound;
	
	cout << "Objects: " << nrOfObjects << ", Transforms: " << nrOfTransformations << endl << endl;
	
	nrOfFiles = nrOfObjects * nrOfTransformations;
}


void ImagePoissonSpikingNeurons::load_filter_parameters(const char * filterParameters, const char * inputDirectory) {


	cout << "Reading filter parameters:" << endl;

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

	int line_index = 0;
	while(getline(filterParametersStream, dirNameBase)) {

		cout << "READING: " << dirNameBase << endl;

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
						dimension = num;
						break; 
				}

			}	
		} 

		line_index++;

	}

	depth = filterWavelengths->size()*filterPhases->size()*filterOrientations->size();
}


void ImagePoissonSpikingNeurons::loadInput(const char * inputDirectory) {

	vector<vector<vector<vector<float> > > > tmp2(nrOfObjects * nrOfTransformations, vector<vector<vector<float> > >(depth, vector<vector<float> >(dimension, vector<float>(dimension)))); 
	*buffer = tmp2;
	
	cout << "inputNames.size: " << inputNames.size() << endl;

	for(unsigned f = 0;f < inputNames.size();f++) {
		
		cout << "Loading Stimuli #" << f << endl;
		
		for(u_short orientation = 0;orientation < filterOrientations->size();orientation++)	{ // Orientations
			for(u_short wavelength = 0;wavelength < filterWavelengths->size();wavelength++) {	// Wavelengths
				for(u_short phase = 0;phase < filterPhases->size();phase++) {				// Phases
					
					// Read input to network
					ostringstream dirStream;

					dirStream << inputDirectory << "Filtered/" << inputNames[f] << ".flt" << "/"
					<< inputNames[f] << '.' << filterWavelengths->at(wavelength) << '.' 
					<< filterOrientations->at(orientation) << '.' << filterPhases->at(phase) << ".gbo";
					
					string t = dirStream.str();
					
					// Open&Read gabor filter file
					fstreamWrapper gaborStream;
					
					try {
						float firing;
						gaborStream.open(t.c_str(), std::ios_base::in | std::ios_base::binary);
						
						// Read flat buffer into 2d slice of V1
						u_short d = mapToV1Depth(orientation,wavelength,phase);
						for(u_short i = 0;i < dimension;i++)
							for(u_short j = 0;j < dimension;j++) {
								
								gaborStream >> firing;
								
								if(firing < 0) {
									cerr << "Negative firing loaded from filter!!!" << endl;
									exit(EXIT_FAILURE);
								}

								// printf("firing: %f\n", firing);
								
								(*buffer)[f][d][i][j] = firing;
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
}

void ImagePoissonSpikingNeurons::copy_buffer_to_device() {
	printf("HEYOO\n");
	CudaSafeCall(cudaMalloc((void **)&d_buffer, sizeof(float)*100));
}


u_short ImagePoissonSpikingNeurons::mapToV1Depth(u_short orientationIndex, u_short wavelengthIndex, u_short phaseIndex) {
	
	return orientationIndex * (filterWavelengths->size() * filterPhases->size()) + wavelengthIndex * filterPhases->size() + phaseIndex;
}


