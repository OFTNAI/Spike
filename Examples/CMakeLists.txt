# if (ENABLE_UBLAS)
# include_directories(${Boost_INCLUDE_DIRS})
foreach(example
    CPUTest
    TSCSOptimiseFourLayerVisionSpikingModelFiringRates
    VogelsAbbotNet
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  #set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_OPENCL")
endforeach()
# endif (ENABLE_UBLAS)
