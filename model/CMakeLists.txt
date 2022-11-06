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

# ==================================================================
add_library(vopat_model
  Brick.h
  Brick.cpp

  StructuredModel.h
  StructuredModel.cpp
  UMeshModel.h
  UMeshModel.cpp

  Model.h
  Model.cpp
  Config.h
  Config.cpp
  )
target_link_libraries(vopat_model
  PUBLIC
  vopat_common
  )

add_custom_target(vopat_umesh_tools ALL)
add_dependencies(vopat_umesh_tools umeshPartitionSpatially)
target_link_libraries(vopat_model PUBLIC
  umesh
  miniScene
  )

# ==================================================================
add_executable(vopatSplitRaw
  splitter.cpp
  )
target_link_libraries(vopatSplitRaw
  vopat_model
  )

# ==================================================================
add_executable(vopatSplitUMesh
  partitionUMesh.cpp
  )
target_link_libraries(vopatSplitUMesh
  vopat_model
  )

# ==================================================================
add_executable(vopatMakeMagField
  makeMagField.cpp
  )
target_link_libraries(vopatMakeMagField
  vopat_model
  )

# ==================================================================
option(BUILD_NC_TO_RAW_TOOL "Build netcdf-to-raw tool" OFF)
if (BUILD_NC_TO_RAW_TOOL)
add_executable(ncToRaw
  ncToRaw.cpp
  )
target_link_libraries(ncToRaw
  vopat_model
  netcdf_c++4
  )
endif()


# ==================================================================
option(BUILD_FITS_TO_RAW_TOOL "Build fits-to-raw tool" OFF)
if (BUILD_FITS_TO_RAW_TOOL)
find_package(cfitsio REQUIRED)
add_executable(fitsToRaw
  fitsToRaw.cpp
  )
target_include_directories(fitsToRaw PUBLIC ${CFITSIO_INCLUDE_DIRS})
target_link_libraries(fitsToRaw
  vopat_model
  ${CFITSIO_LIBRARIES}
  )
endif()


