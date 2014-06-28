/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkVTKPointCloudReader.hxx,v $
  Language:  C++
  Date:      $Date: 2011-10-05 18:01:00 $
  Version:   $Revision: 1.19 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVTKPointCloudReader_hxx
#define __itkVTKPointCloudReader_hxx

#include <fstream>
#include <cstdio>
#include <cstring>

#include "itkVTKPointCloudReader.h"

// VTK includes
#include "vtkCellArray.h"
#include "vtkNew.h"
#include "vtkPoints.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkPolyData.h"
#include <vtkDataReader.h>
#include <vtkDataSetReader.h>
#include <vtkGenericDataObjectReader.h>

namespace itk
{


//
// Constructor
//
template<class TOutputMesh>
VTKPointCloudReader<TOutputMesh>
::VTKPointCloudReader()
{
  //
  // Create the output
  //
  typename TOutputMesh::Pointer output = TOutputMesh::New();
  this->ProcessObject::SetNumberOfRequiredOutputs(1);
  this->ProcessObject::SetNthOutput(0, output.GetPointer());
}

//
// Destructor
//
template<class TOutputMesh>
VTKPointCloudReader<TOutputMesh>
::~VTKPointCloudReader()
{
}

template<class TOutputMesh>
void
VTKPointCloudReader<TOutputMesh>
::GenerateData()
{

  OutputMeshType * outputMesh = this->GetOutput();

  if( m_FileName == "" )
  {
    itkExceptionMacro(<< "No input FileName");
  }

  vtkNew<vtkGenericDataObjectReader> reader;
  reader->SetFileName(m_FileName.c_str());
  reader->Update();

  vtkPointSet *vtkMesh;
  if(reader->IsFilePolyData())
  {
    vtkMesh = vtkPolyData::SafeDownCast(reader->GetPolyDataOutput());
  }
  else
  {
    vtkMesh = vtkUnstructuredGrid::SafeDownCast(reader->GetUnstructuredGridOutput());
  }

  PointIdentifier numberOfPoints = vtkMesh->GetNumberOfPoints();
  PointsContainer * points = outputMesh->GetPoints();
  points->Reserve( numberOfPoints );

  //
  // Load the point coordinates into the itk::Mesh
  //
  PointType point;

  for( PointIdentifier pointId = 0; pointId < numberOfPoints; ++pointId )
  {
    double *p = vtkMesh->GetPoint(pointId);
    point[0] = p[0];
    point[1] = p[1];
    point[2] = p[2];

    outputMesh->SetPoint( pointId, point );
  }

}

template<class TOutputMesh>
void
VTKPointCloudReader<TOutputMesh>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "FileName: " << m_FileName << std::endl;
  os << indent << "Version: " << m_Version << std::endl;
  os << indent << "Header: " << m_Header << std::endl;
}

} //end of namespace itk


#endif
