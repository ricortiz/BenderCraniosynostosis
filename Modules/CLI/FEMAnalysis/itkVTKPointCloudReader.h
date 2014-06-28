/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkVTKPointCloudReader.h,v $
  Language:  C++
  Date:      $Date: 2011-10-05 18:01:00 $
  Version:   $Revision: 1.9 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVTKPointCloudReader_h
#define __itkVTKPointCloudReader_h

#include "itkIntTypes.h"
#include "itkMesh.h"
#include "itkMeshSource.h"
#include "itkTetrahedronCell.h"


namespace itk
{

 /** \class VTKPointCloudReader
 * \brief
 * Reads a VTKUnstructuredGrid or or VTKPolyData file and create an itkMesh.
 *
 * Caveat1: VTKPointCloudReader can only read point clouds.
 * Caviet2: VTKPointCloudReader can only read vtk legacy files.
 * Caveat3: VTKPointCloudReader cannot read binary vtk files.
 */
template <class TOutputMesh>
class VTKPointCloudReader : public MeshSource<TOutputMesh>
{
public:
  /** Standard "Self" typedef. */
  typedef VTKPointCloudReader  Self;
  typedef MeshSource<TOutputMesh>   Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VTKPointCloudReader, MeshSource);

  /** Hold on to the type information specified by the template parameters. */
  typedef TOutputMesh                          OutputMeshType;
  typedef typename OutputMeshType::MeshTraits  MeshTraits;
  typedef typename OutputMeshType::PointType   PointType;
  typedef typename MeshTraits::PixelType       PixelType;

  /** Some convenient typedefs. */
  typedef typename OutputMeshType::Pointer         OutputMeshPointer;
  typedef typename OutputMeshType::PointIdentifier PointIdentifier;

  typedef typename OutputMeshType::PointsContainerPointer
    PointsContainerPointer;

  typedef typename OutputMeshType::PointsContainer
    PointsContainer;

  /** Set the resolution level to be used for generating cells in the
   * Sphere. High values of this parameter will produce sphere with more
   * triangles. */
  /** Set/Get the name of the file to be read. */
  itkSetStringMacro(FileName);
  itkGetStringMacro(FileName);

  /** Get the file version line */
  itkGetStringMacro(Version);

  /** Get the file header line */
  itkGetStringMacro(Header);

protected:
  VTKPointCloudReader();
  ~VTKPointCloudReader();
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Reads the file */
  void GenerateData();

private:
  VTKPointCloudReader(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  /** Filename to read */
  std::string m_FileName;
  std::string m_Header;
  std::string m_Version;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVTKPointCloudReader.hxx"
#endif

#endif //_itkVTKPointCloudReader_h
