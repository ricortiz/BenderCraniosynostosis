#============================================================================
#
# Program: Bender
#
# Copyright (c) Kitware Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#============================================================================

#add_subdirectory(Scripts)
set(modules
  BenderLabelStatistics
  BenderSampleData
  BenderWelcome
  SkinModelMaker
  Workflow
  SimpleWorkflow
  )

set(Bender_MODULES)

foreach(module ${modules})
  list(APPEND Bender_MODULES
    ${CMAKE_CURRENT_SOURCE_DIR}/${module})
endforeach()

set_property(GLOBAL APPEND PROPERTY Bender_MODULES ${Bender_MODULES})
