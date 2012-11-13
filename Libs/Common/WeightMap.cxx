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

#include "WeightMap.h"
#include <itkImageRegionIterator.h>
#include <iostream>

WeightMap::WeightMap():Cols(0)
{
}
void WeightMap::Init(const std::vector<Voxel>& voxels, const Region& region)
{
  this->Cols = voxels.size();
  this->RowSize.resize(this->Cols,0);

  this->LUTIndex = WeightLUTIndex::New();
  this->LUTIndex->SetRegions(region);
  this->LUTIndex->Allocate();

  itk::ImageRegionIterator<WeightLUTIndex> it(this->LUTIndex,region);
  for(it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
    it.Set(std::numeric_limits<std::size_t>::max());
    }

  for(size_t j=0; j<voxels.size(); j++)
    {
    this->LUTIndex->SetPixel(voxels[j], j);
    }
}

bool WeightMap::Insert(const Voxel& v, SiteIndex index, float value)
{
  if(value<=0)
    {
    return false;
    }
  size_t j = this->LUTIndex->GetPixel(v);
  assert(j<Cols);
  size_t i = this->RowSize[j];
  if (i>=LUT.size())
    {
    this->AddRow();
    }
  WeightEntry& weight = LUT[i][j];
  weight.Index = index;
  weight.Value = value;

  this->RowSize[j]++;
  return true;
}

void WeightMap::Get(const Voxel& v, WeightVector& values) const
{
  values.Fill(0);
  size_t j = this->LUTIndex->GetPixel(v);
  assert(j<Cols);

  size_t rows = this->RowSize[j];

  for(size_t i=0; i<rows; i++)
    {
    const WeightEntry& entry = LUT[i][j];
    values[entry.Index] = entry.Value;

    }
}

void WeightMap::AddRow()
{
  LUT.push_back(WeightEntries());
  WeightEntries& newRow = LUT.back();
  newRow.resize(Cols);
}

void WeightMap::Print()
{
  int numEntries(0);
  for(RowSizes::iterator r=this->RowSize.begin();r!=this->RowSize.end();r++)
    {
    numEntries+=*r;
    }
  std::cout<<"Weight map "<<LUT.size()<<"x"<<Cols<<" has "<<numEntries<<" entries"<<std::endl;
}




