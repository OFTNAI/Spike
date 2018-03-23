add_subdirectory(Dummy)
if(BUILD_WITH_CUDA)
  add_subdirectory(CUDA)
endif()

add_library (Backend OBJECT
  Context.cpp
  )
