# if (ENABLE_UBLAS)
# include_directories(${Boost_INCLUDE_DIRS})
foreach(experiment
    CPUTest
    )
  add_executable(${experiment} ${experiment}.cpp)
  target_link_libraries(${experiment} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_OPENCL")
endforeach()
# endif (ENABLE_UBLAS)
