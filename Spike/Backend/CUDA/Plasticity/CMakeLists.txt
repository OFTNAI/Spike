CUDA_ADD_LIBRARY (SpikeCUDAPlasticity
  Plasticity.cu
  STDPPlasticity.cu
  EvansSTDPPlasticity.cu
  MasquelierSTDPPlasticity.cu
  vanRossumSTDPPlasticity.cu
  VogelsSTDPPlasticity.cu
  STATIC)
