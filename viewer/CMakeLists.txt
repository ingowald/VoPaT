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
find_package(Qt5Core)
find_package(Qt5Gui)
find_package(Qt5Widgets)
qt5_wrap_cpp(WRAPPED_HEADERS isoDialog.h)
qt5_wrap_ui(WRAPPED_UI isoDialog.ui)
add_executable(vopatQTViewer
  AppInterface.cpp
  AppInterface.h
  qtViewer.cpp
  isoDialog.cpp
  ${WRAPPED_UI}
  ${WRAPPED_HEADERS}
  )
# moc'd headers:
target_include_directories(vopatQTViewer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(vopatQTViewer
  PUBLIC
  vopat_render
  qtOWL
  )
set_property(TARGET vopatQTViewer PROPERTY CUDA_ARCHITECTURES "70;75;80;86") 
set_property(TARGET vopatQTViewer PROPERTY POSITION_INDEPENDENT_CODE ON) 


# ==================================================================
set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
add_executable(vopatHeadlessViewer
  qtViewer.cpp
  AppInterface.cpp
  AppInterface.h
  headless.cpp
  )
target_link_libraries(vopatHeadlessViewer
  PUBLIC
  vopat_render
  qtOWL
  )
set_property(TARGET vopatHeadlessViewer PROPERTY CUDA_ARCHITECTURES "70;75;80;86") 
set_property(TARGET vopatHeadlessViewer PROPERTY POSITION_INDEPENDENT_CODE ON) 
target_compile_definitions(vopatHeadlessViewer PUBLIC HEADLESS)

