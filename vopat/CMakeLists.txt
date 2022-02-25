# ======================================================================== #
# Copyright 2022-2022 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(MPI REQUIRED)

# ==================================================================
add_library(vopat_model
  common.h
  model/Brick.h
  model/Brick.cpp
  model/Model.h
  model/Model.cpp
  )
target_include_directories(vopat_model PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(vopat_model PUBLIC
  owl::owl
  )

# ==================================================================
add_executable(vopatSplitRaw
  model/splitter.cpp
  )
target_link_libraries(vopatSplitRaw
  vopat_model
  )

#add_executable(vopatGenMacrocells
#  model/macrocells.cpp
#  )
#target_link_libraries(vopatGenMacrocells
#  vopat_model
#  )

add_library(vopat_render
  Renderer.cpp
  
  render/DistributedRendererBase.h
  render/DistributedRendererBase.cu
  render/RayForwardingRenderer.h
  render/RayForwardingRenderer.cu
#  render/SimpleNodeRenderer.cu
  render/Woodcock.cu
  render/CellMarchRenderer.cu
  render/StepMarchRenderer.cu
  mpi/MPIBackend.cpp
  mpi/MPIWorker.cpp
  mpi/MPIMaster.cpp
  )
target_link_libraries(vopat_render
  PUBLIC
  vopat_model
  owl_viewer
  MPI::MPI_CXX
  )
set_property(TARGET vopat_render PROPERTY CUDA_ARCHITECTURES "70;75;80;86") 

# ==================================================================
add_executable(vopatViewer
  viewer.cpp
  )
target_link_libraries(vopatViewer
  PUBLIC
  vopat_render
  )
#set_property(TARGET vopatViewer PROPERTY CUDA_ARCHITECTURES "70;75;80;86") 


# ==================================================================
set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
add_executable(vopatQTViewer
  qtViewer.cpp
  )
target_link_libraries(vopatQTViewer
  PUBLIC
  vopat_render
  qtOWL
  )
set_property(TARGET vopatQTViewer PROPERTY CUDA_ARCHITECTURES "70;75;80;86") 
set_property(TARGET vopatQTViewer PROPERTY POSITION_INDEPENDENT_CODE ON) 

#get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
#string(REPLACE "-fPIC" "" new_core_options ${core_options})
#set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
#set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)



#get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
#string(REPLACE "-fPIC" "" new_core_options ${core_options})
#set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
#set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

