#ifndef ImagePoissonInputSpikingNeurons_H
#define ImagePoissonInputSpikingNeurons_H

// #define SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP

#include "PoissonInputSpikingNeurons.hpp"

#include <vector>
#include <string>

// using namespace std;

struct image_poisson_input_spiking_neuron_parameters_struct : poisson_input_spiking_neuron_parameters_struct {
	image_poisson_input_spiking_neuron_parameters_struct(): gabor_index(-1) { poisson_input_spiking_neuron_parameters_struct(); }

	int gabor_index;
};

class ImagePoissonInputSpikingNeurons; // forward definition

namespace Backend {
  class ImagePoissonInputSpikingNeurons : public virtual PoissonInputSpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(ImagePoissonInputSpikingNeurons);
    virtual void copy_rates_to_device() = 0;
  };
}

class ImagePoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
public:
  // Constructor/Destructor
  ImagePoissonInputSpikingNeurons();
  ~ImagePoissonInputSpikingNeurons() override;

  SPIKE_ADD_BACKEND_GETSET(ImagePoissonInputSpikingNeurons, PoissonInputSpikingNeurons);
  void init_backend(Context* ctx = _global_ctx) override;
  
  int total_number_of_objects = 0;
  int total_number_of_transformations_per_object = 0;
  bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index); // Moved from Input Neuron Class
  
  int AddGroup(neuron_parameters_struct * group_params) override;
  void AddGroupForEachGaborType(neuron_parameters_struct * group_params);

  void state_update(float current_time_in_seconds, float timestep) override;

  void set_up_rates(const char * fileList, const char * filterParameters, const char * inputDirectory, float max_rate_scaling_factor);
  void load_image_names_from_file_list(const char * fileList, const char * inputDirectory);
  void load_gabor_filter_parameters(const char * filterParameters, const char * inputDirectory);
  void load_rates_from_files(const char * inputDirectory, float max_rate_scaling_factor);
  void copy_rates_to_device();
  int calculate_gabor_index(int orientationIndex, int wavelengthIndex, int phaseIndex);

  //JI VARIABLES
  float * gabor_input_rates = nullptr;

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

private:
  std::shared_ptr<::Backend::ImagePoissonInputSpikingNeurons> _backend;
};

#endif
