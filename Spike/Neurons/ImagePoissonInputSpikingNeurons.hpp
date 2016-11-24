#ifndef ImagePoissonInputSpikingNeurons_H
#define ImagePoissonInputSpikingNeurons_H

#include "PoissonInputSpikingNeurons.hpp"

#include <vector>
#include <string>

// using namespace std;

struct image_poisson_input_spiking_neuron_parameters_struct : poisson_input_spiking_neuron_parameters_struct {
	image_poisson_input_spiking_neuron_parameters_struct(): gabor_index(-1) { poisson_input_spiking_neuron_parameters_struct(); }

	int gabor_index;
};

namespace Backend {
  class ImagePoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
  public:
  };
}

#include "Spike/Backend/Dummy/Neurons/ImagePoissonInputSpikingNeurons.hpp"

class ImagePoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
public:
  // Constructor/Destructor
  ImagePoissonInputSpikingNeurons();
  ~ImagePoissonInputSpikingNeurons();

  ADD_BACKEND_GETTER(ImagePoissonInputSpikingNeurons);
  
  virtual int AddGroup(neuron_parameters_struct * group_params);
  void AddGroupForEachGaborType(neuron_parameters_struct * group_params);

  // TODO:
  // virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);

  void set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory, float max_rate_scaling_factor);
  void load_image_names_from_file_list(const char * fileList, const char * inputDirectory);
  void load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory);
  void load_rates_from_files(const char * inputDirectory, float max_rate_scaling_factor);
  virtual void copy_rates_to_device();
  int calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex);

  virtual void prepare_backend(Context* ctx = _global_ctx);
  
  //JI VARIABLES
  float * gabor_input_rates = NULL;

  int total_number_of_phases = 0;
  int total_number_of_wavelengths = 0;
  int total_number_of_orientations = 0;
  int image_width = 0;

  int total_number_of_rates = 0;
  int total_number_of_rates_per_image = 0;

  int total_number_of_gabor_types = 0;

  //OLD VARIABLES
  std::vector<std::string> inputNames;

  std::vector<float> * filterPhases;
  std::vector<int>  * filterWavelengths;
  std::vector<float> * filterOrientations;
	
};

#endif
