CUDA_ADD_LIBRARY (SpikeCUDANeurons
  Neurons.cu
  SpikingNeurons.cu
  InputSpikingNeurons.cu
  GeneratorInputSpikingNeurons.cu
  PoissonInputSpikingNeurons.cu
  ImagePoissonInputSpikingNeurons.cu
  LIFSpikingNeurons.cu
  AdExSpikingNeurons.cu
  IzhikevichSpikingNeurons.cu
  STATIC)
