/*=========================================================================

  Program: Bender

  Copyright (c) Kitware Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0.txt

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/

#ifndef LABELMAPFIELD_H
#define LABELMAPFIELD_H

// Cleaver includes
#include <Cleaver/ScalarField.h>
#include <Cleaver/BoundingBox.h>

// ITK includes
#include <itkImage.h>
#include <itkLinearInterpolateImageFunction.h>

namespace Cleaver {

class LabelMapField :  public ScalarField
{
public:
  typedef float PixelType;
  typedef itk::Image<PixelType,3> ImageType;
  typedef itk::LinearInterpolateImageFunction<ImageType> InterpolationType;

public:
  LabelMapField(ImageType::Pointer LabelImage);
  ~LabelMapField();
  virtual float valueAt(float x, float y, float z) const;
  virtual float valueAt(const vec3 &x) const;

  virtual BoundingBox bounds() const;
  BoundingBox dataBounds() const;
  void SetGenerateDataFromLabels(bool _generateData);
  bool GetGenerateDataFromLabels() {
    return this->GenerateDataFromLabels;
  }

private:
  ImageType::Pointer         LabelMap;
  InterpolationType::Pointer Interpolant;
  BoundingBox                Bounds;
  vec3                       Scale;
  float *                    data;
  bool                       GenerateDataFromLabels;
};

} // namespace

#endif // LABELMAPFIELD_H
