set (CUDANeurons
  ${CMAKE_CURRENT_SOURCE_DIR}/Neurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/SpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/InputSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/GeneratorInputSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/PoissonInputSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/GeneralPoissonInputSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/ImagePoissonInputSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/AdExSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/LIFSpikingNeurons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichSpikingNeurons.cu
  PARENT_SCOPE)
