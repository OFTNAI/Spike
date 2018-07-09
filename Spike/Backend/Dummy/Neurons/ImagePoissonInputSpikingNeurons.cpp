#include "ImagePoissonInputSpikingNeurons.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Dummy, ImagePoissonInputSpikingNeurons);

namespace Backend {
  namespace Dummy {
    void ImagePoissonInputSpikingNeurons::prepare() {
      PoissonInputSpikingNeurons::prepare();
    }
      
    void ImagePoissonInputSpikingNeurons::reset_state() {
      PoissonInputSpikingNeurons::reset_state();
    }

    void ImagePoissonInputSpikingNeurons::copy_rates_to_device() {
    }
  }
}
