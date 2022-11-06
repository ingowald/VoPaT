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

cmake_policy(SET CMP0048 NEW)
project(DPVoPaT VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/)
include(cmake/configure_build_type.cmake)

add_subdirectory(submodules/umesh EXCLUDE_FROM_ALL)

add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
target_compile_definitions(owl_viewer
  PUBLIC
  -DOWL_FORCE_SLOW_DISPLAY=1
  )

add_subdirectory(submodules/cuteeOWL EXCLUDE_FROM_ALL)
target_compile_definitions(qtOWL
  PUBLIC
  -DCUTEEOWL_FORCE_SLOW_DISPLAY=1
  )

add_subdirectory(submodules/miniScene EXCLUDE_FROM_ALL)

option(VOPAT_VOXELS_AS_TEXTURE "Use a CUDA texture to access voxels on the device" ON)

#set(CMAKE_CUDA_ARCHITECTURES "75") 
add_subdirectory(common)
add_subdirectory(model)
add_subdirectory(vopat)
add_subdirectory(viewer)

