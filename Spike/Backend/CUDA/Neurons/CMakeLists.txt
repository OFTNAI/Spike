CUDA_ADD_LIBRARY (SpikeCUDANeurons
  Neurons.cu
  SpikingNeurons.cu
  InputSpikingNeurons.cu
  GeneratorInputSpikingNeurons.cu
  PoissonInputSpikingNeurons.cu
  GeneralPoissonInputSpikingNeurons.cu
  ImagePoissonInputSpikingNeurons.cu
  AdExSpikingNeurons.cu
  LIFSpikingNeurons.cu
  IzhikevichSpikingNeurons.cu
  STATIC)
