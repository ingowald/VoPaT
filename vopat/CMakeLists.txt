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

embed_ptx(
  OUTPUT_TARGET
  deviceCode-ptx
  
  PTX_LINK_LIBRARIES
#  owl::owl_static
  umesh
  miniScene
  vopat_common
  
  SOURCES
  deviceCode.cu
)


add_library(vopat_render
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/StructuredVolume.cu
  volume/UMeshVolume.cu
  volume/Volume.cpp

  surface/MeshGeom.cpp

#  Renderer.cpp
#  ModelConfig.cpp
  
  NextDomainKernel.h
  NextDomainKernel.cu
  
#  Volume.cu
  
#  DistributedRendererBase.h
#  DistributedRendererBase.cu
  AddLocalFBsLayer.h
  AddLocalFBsLayer.cu
  ForwardingLayer.h
  ForwardingLayer.cu
  
  VopatRenderer.h
  VopatRenderer.cu
  
  Woodcock.h
  Woodcock.cu
#  IntentionallyWrong.cu
#  CellMarchRenderer.cu
)
if (VOPAT_USE_RAFI)
  target_compile_definitions(deviceCode-ptx PUBLIC -DVOPAT_USE_RAFI=1)
  target_compile_definitions(deviceCode-ptx_ptx PUBLIC -DVOPAT_USE_RAFI=1)
  target_link_libraries(deviceCode-ptx PUBLIC rafi)
  target_link_libraries(deviceCode-ptx_ptx PUBLIC rafi)
endif()


target_include_directories(deviceCode-ptx PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(deviceCode-ptx_ptx PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(vopat_render
  PUBLIC
  vopat_common
  vopat_model
#  owl_viewer
  deviceCode-ptx
  )
#set_property(TARGET vopat_render PROPERTY CUDA_ARCHITECTURES "80")
#set_property(TARGET vopat_render PROPERTY CUDA_ARCHITECTURES "70;75;80;86")

#target_compile_options(vopat_render PUBLIC
#                        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>
#                        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
#                        )

if(VOPAT_VOXELS_AS_TEXTURE)
  target_compile_definitions(vopat_render PUBLIC VOPAT_VOXELS_AS_TEXTURE=1)
endif()

