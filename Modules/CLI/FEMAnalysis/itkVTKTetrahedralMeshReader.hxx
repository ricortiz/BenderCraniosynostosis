/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkVTKTetrahedralMeshReader.hxx,v $
  Language:  C++
  Date:      $Date: 2011-10-05 18:01:00 $
  Version:   $Revision: 1.19 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVTKTetrahedralMeshReader_hxx
#define __itkVTKTetrahedralMeshReader_hxx

#include <fstream>
#include <cstdio>
#include <cstring>

#include "itkVTKTetrahedralMeshReader.h"

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
#include <vtkSmartPointer.h>

namespace itk
{


//
// Constructor
//
template<class TOutputMesh>
VTKTetrahedralMeshReader<TOutputMesh>
::VTKTetrahedralMeshReader()
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
VTKTetrahedralMeshReader<TOutputMesh>
::~VTKTetrahedralMeshReader()
{
}

template<class TOutputMesh>
void
VTKTetrahedralMeshReader<TOutputMesh>
::GenerateData()
{

  OutputMeshType * outputMesh = this->GetOutput();

  outputMesh->SetCellsAllocationMethod(
    OutputMeshType::CellsAllocatedDynamicallyCellByCell );


  if( m_FileName == "" )
  {
    itkExceptionMacro(<< "No input FileName");
  }

  vtkNew<vtkGenericDataObjectReader> reader;
  reader->SetFileName(m_FileName.c_str());
  reader->Update();

  vtkSmartPointer<vtkPointSet> vtkMesh;
  if(reader->IsFilePolyData())
  {
    vtkMesh = vtkPolyData::SafeDownCast(reader->GetPolyDataOutput());
  }
  if(reader->IsFileUnstructuredGrid())
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

  CellIdentifier numberOfCells = vtkMesh->GetNumberOfCells();
  vtkCellArray *tetras;
  if(reader->IsFilePolyData())
    tetras = vtkPolyData::SafeDownCast(vtkMesh)->GetPolys();
  else
    tetras = vtkUnstructuredGrid::SafeDownCast(vtkMesh)->GetCells();
  //
  // Load the cells into the itk::Mesh
  //

  tetras->InitTraversal();
  PointIdentifier numberOfCellPoints;

  vtkNew<vtkIdList> element;
  for (CellIdentifier cellId = 0; tetras->GetNextCell(element.GetPointer()); ++cellId)
  {
    numberOfCellPoints = element->GetNumberOfIds();
    if(element->GetNumberOfIds() != 4)
    {
      itkExceptionMacro(<< "Error reading file: " << m_FileName
      << "\nnumberOfCellPoints != 4\n"
      << "numberOfCellPoints= " << numberOfCellPoints
      << ". VTKTetrahedralMeshReader can only read tetrahedra");
    }

    vtkIdType ids[4] = {
      element->GetId(0),
      element->GetId(1),
      element->GetId(2),
      element->GetId(3)
    };

    if( ids[0] < 0 || ids[1] < 0 || ids[2] < 0 || ids[3] < 0 )
    {
      itkExceptionMacro(<< "Error reading file: " << m_FileName
      << "point ids must be >= 0.\n"
      "ids=" << ids[0] << " " << ids[1] << " " << ids[2] << " " << ids[3]);
    }

    if( static_cast<PointIdentifier>( ids[0] ) >= numberOfPoints ||
      static_cast<PointIdentifier>( ids[1] ) >= numberOfPoints ||
      static_cast<PointIdentifier>( ids[2] ) >= numberOfPoints ||
      static_cast<PointIdentifier>( ids[3] ) >= numberOfPoints )
    {
      itkExceptionMacro(<< "Error reading file: " << m_FileName
      << "Point ids must be < number of points: "
      << numberOfPoints
      << "\nids= " << ids[0] << " " << ids[1] << " " << ids[2] << " " << ids[3]);
    }

    CellAutoPointer cell;

    TetrahedronCellType * tetrahedronCell = new TetrahedronCellType;
    for( PointIdentifier pointId = 0; pointId < numberOfCellPoints; pointId++ )
    {
      tetrahedronCell->SetPointId( pointId, ids[pointId] );
    }

    cell.TakeOwnership( tetrahedronCell );
    outputMesh->SetCell( cellId, cell );
  }

}

template<class TOutputMesh>
void
VTKTetrahedralMeshReader<TOutputMesh>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "FileName: " << m_FileName << std::endl;
  os << indent << "Version: " << m_Version << std::endl;
  os << indent << "Header: " << m_Header << std::endl;
}

} //end of namespace itk


#endif
