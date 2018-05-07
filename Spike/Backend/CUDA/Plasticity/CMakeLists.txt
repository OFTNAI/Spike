set (CUDAPlasticity
  ${CMAKE_CURRENT_SOURCE_DIR}/Plasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/STDPPlasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/EvansSTDPPlasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/MasquelierSTDPPlasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/vanRossumSTDPPlasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/VogelsSTDPPlasticity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/WeightNormSTDPPlasticity.cu
  PARENT_SCOPE)
