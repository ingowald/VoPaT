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

add_library(vopat_common
  STATIC
  vopat.h
  
  mpi/MPIBackend.cpp
#  mpi/MPIWorker.cpp
#  mpi/MPIMaster.cpp
  )

target_link_libraries(vopat_common
  PUBLIC
  owl::owl_static
#  MPI::MPI_CXX
  MPI::MPI_C
  )

target_include_directories(vopat_common
  PUBLIC
  ${PROJECT_SOURCE_DIR}
)

if (VOPAT_USE_RAFI)
  target_compile_definitions(vopat_common PUBLIC -DVOPAT_USE_RAFI=1)
  target_link_libraries(vopat_common PUBLIC rafi)
endif()



