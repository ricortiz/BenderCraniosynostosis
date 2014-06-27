/*=========================================================================
 *
 *   Program: Bender
 *
 *   Copyright (c) Kitware Inc.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 *   =========================================================================*/

// Bender includes
#include "ICPMeshRegistrationCLP.h"
#include "benderIOUtils.h"
#include "itkVTKTetrahedralMeshReader.h"

// ITK include
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkMesh.h>
#include <itkTetrahedronCell.h>
#include <itkVector.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkTranslationTransform.h>
#include <itkEuclideanDistancePointMetric.h>
#include <itkNormalizedCorrelationPointSetToImageMetric.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkPointSetToPointSetRegistrationMethod.h>
#include <itkPointSetToImageRegistrationMethod.h>
#include <itkDanielssonDistanceMapImageFilter.h>
#include <itkPointSetToImageFilter.h>
#include <itkTransformFileWriter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVTKPolyDataWriter.h>
#include <itkMeshFileWriter.h>
#include <itkMeshFileReader.h>

// VTK includes
#include <vtkCellArray.h>
#include <vtkCellCenters.h>
#include <vtkCellData.h>
#include <vtkCellDataToPointData.h>
#include <vtkDataArray.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkThreshold.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkMath.h>
#include <vtkTriangleFilter.h>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};
public:

  typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
  typedef const OptimizerType *                OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    OptimizerPointer optimizer =
    dynamic_cast< OptimizerPointer >( object );

    if( ! itk::IterationEvent().CheckEvent( &event ) )
    {
      return;
    }

    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;

  }

};

itk::TranslationTransform< double,3>::Pointer
registerMeshes(itk::PointSet<float,3> *fixedMesh,
               itk::PointSet<float,3> *movingMesh,
               const int &numberOfIterations,
               const double &gradientTolerance,
               const double &valueTolerance,
               const double &epsilonFunction,
               bool Verbose = false)
{
  typedef itk::PointSet<float,3> PointSetType;
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Number of moving Points = " << movingMesh->GetNumberOfPoints( ) << std::endl;
  }
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Number of fixed Points = " << fixedMesh->GetNumberOfPoints() << std::endl;
  }
  //-----------------------------------------------------------
  // Set up  the Metric
  //-----------------------------------------------------------
  typedef itk::EuclideanDistancePointMetric<PointSetType,PointSetType> MetricType;
  MetricType::Pointer  metric = MetricType::New();
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Set up a Transform..." << std::endl;
  }
  //-----------------------------------------------------------
  // Set up a Transform
  //-----------------------------------------------------------
  typedef itk::TranslationTransform< double, 3> TransformType;
  TransformType::Pointer transform = TransformType::New();

  // Optimizer Type
  typedef itk::LevenbergMarquardtOptimizer OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetUseCostFunctionGradient(false);

  // Registration Method
  typedef itk::PointSetToPointSetRegistrationMethod<PointSetType,PointSetType> RegistrationType;
  RegistrationType::Pointer   registration  = RegistrationType::New();

  // Scale the translation components of the Transform in the Optimizer
  OptimizerType::ScalesType scales( transform->GetNumberOfParameters() );
  scales.Fill( 0.01 );

  optimizer->SetScales( scales );
  optimizer->SetNumberOfIterations( numberOfIterations );
  optimizer->SetValueTolerance( valueTolerance );
  optimizer->SetGradientTolerance( gradientTolerance );
  optimizer->SetEpsilonFunction( epsilonFunction );

  // Start from an Identity transform (in a normal case, the user
  // can probably provide a better guess than the identity...
  transform->SetIdentity();
  registration->SetInitialTransformParameters( transform->GetParameters() );

  //------------------------------------------------------
  // Connect all the components required for Registration
  //------------------------------------------------------
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetTransform(     transform     );
  registration->SetFixedPointSet( fixedMesh );
  registration->SetMovingPointSet(   movingMesh   );
  registration->SetNumberOfThreads(5);
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Prepare the Distance Map..." << std::endl;
  }
  //------------------------------------------------------
  // Prepare the Distance Map in order to accelerate
  // distance computations.
  //------------------------------------------------------
  //
  //  First map the Fixed Points into a binary image.
  //  This is needed because the DanielssonDistance
  //  filter expects an image as input.
  //
  //-------------------------------------------------
  typedef itk::Image< unsigned char,  3 >  BinaryImageType;

  typedef itk::PointSetToImageFilter<PointSetType,BinaryImageType> PointsToImageFilterType;
  typedef itk::BoundingBox<itk::PointSet<float,3>::PointIdentifier,3,float,itk::PointSet<float,3>::PointsContainer> BBox;
  BBox::Pointer boundingBox = BBox::New();
  boundingBox->SetPoints(fixedMesh->GetPoints());
  boundingBox->ComputeBoundingBox();

  PointsToImageFilterType::Pointer pointsToImageFilter = PointsToImageFilterType::New();
  pointsToImageFilter->SetInput( fixedMesh );
  BinaryImageType::SpacingType spacing;
  spacing.Fill( 1.0 );
  pointsToImageFilter->SetSpacing( spacing );
  pointsToImageFilter->SetOrigin( boundingBox->GetCenter() );
  pointsToImageFilter->Update();
  BinaryImageType::Pointer binaryImage = pointsToImageFilter->GetOutput();
  typedef itk::Image< unsigned short, 3 >  DistanceImageType;
  typedef itk::DanielssonDistanceMapImageFilter<BinaryImageType, DistanceImageType> DistanceFilterType;
  DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
  distanceFilter->SetInput( pointsToImageFilter->GetOutput() );
  distanceFilter->Update();
  metric->SetDistanceMap( distanceFilter->GetOutput() );
  // Connect an observer
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

    itk::ImageFileWriter<BinaryImageType>::Pointer w1 = itk::ImageFileWriter<BinaryImageType>::New();
    w1->SetFileName("./binaryImage.mha");
    w1->SetInput(pointsToImageFilter->GetOutput());
    w1->Update();
    itk::ImageFileWriter<DistanceImageType>::Pointer w2 = itk::ImageFileWriter<DistanceImageType>::New();
    w2->SetFileName("./distanceMap.mha");
    w2->SetInput(distanceFilter->GetOutput());
    w2->Update();

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Execute registration..." << std::endl;
  }
  try
  {
    registration->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cout << e << std::endl;
  }

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Finished registration..." << std::endl;
  }
  return transform;
}


itk::TranslationTransform< double,3>::Pointer
registerMeshesAlt(itk::PointSet<float,3> *fixedMesh,
               itk::PointSet<float,3> *movingMesh,
               const int &numberOfIterations,
               const double &gradientTolerance,
               const double &valueTolerance,
               const double &epsilonFunction,
               bool Verbose = false)
{
  typedef itk::PointSet<float,3> PointSetType;
  typedef itk::Image< unsigned short, 3 >  DistanceImageType;
  typedef itk::Image< unsigned char,  3 >  BinaryImageType;
  typedef itk::PointSetToImageFilter<PointSetType,BinaryImageType> PointsToImageFilterType;
  typedef itk::NormalizedCorrelationPointSetToImageMetric<PointSetType,DistanceImageType> MetricType;
  typedef itk::LinearInterpolateImageFunction<DistanceImageType,double> InterpolatorType;
  typedef itk::PointSetToImageRegistrationMethod<PointSetType,DistanceImageType> RegistrationType;
  typedef itk::DanielssonDistanceMapImageFilter<BinaryImageType, DistanceImageType> DistanceFilterType;
  typedef itk::TranslationTransform<double,3> TransformType;
  typedef itk::RegularStepGradientDescentOptimizer   OptimizerType;

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Number of moving Points = " << movingMesh->GetNumberOfPoints( ) << std::endl;
  }
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Number of fixed Points = " << fixedMesh->GetNumberOfPoints() << std::endl;
  }
  //-----------------------------------------------------------
  // Set up  the Metric
  //-----------------------------------------------------------
  MetricType::Pointer  metric = MetricType::New();
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Set up a Transform..." << std::endl;
  }
  //-----------------------------------------------------------
  // Set up a Interpolator
  //-----------------------------------------------------------
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  //-----------------------------------------------------------
  // Set up a Transform
  //-----------------------------------------------------------
  TransformType::Pointer transform = TransformType::New();

  // Optimizer Type
  OptimizerType::Pointer optimizer = OptimizerType::New();

  // Registration Method
  RegistrationType::Pointer   registration  = RegistrationType::New();

  optimizer->SetNumberOfIterations( numberOfIterations );
  optimizer->SetMaximumStepLength( 1.00 );
  optimizer->SetMinimumStepLength( 0.001 );
  optimizer->SetRelaxationFactor( 0.90 );
  optimizer->SetGradientMagnitudeTolerance( 0.05 );
  optimizer->MinimizeOn();

  // Start from an Identity transform (in a normal case, the user
  // can probably provide a better guess than the identity...
  transform->SetIdentity();
  registration->SetInitialTransformParameters( transform->GetParameters() );

  //------------------------------------------------------
  // Connect all the components required for Registration
  //------------------------------------------------------
  registration->SetMetric(        metric        );
  registration->SetOptimizer(     optimizer     );
  registration->SetTransform(     transform     );
  registration->SetInterpolator( interpolator );
  registration->SetFixedPointSet( fixedMesh );

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Prepare the Distance Map..." << std::endl;
  }
  //------------------------------------------------------
  // Prepare the Distance Map in order to accelerate
  // distance computations.
  //------------------------------------------------------
  //
  //  First map the Fixed Points into a binary image.
  //  This is needed because the DanielssonDistance
  //  filter expects an image as input.
  //
  //-------------------------------------------------
  typedef itk::BoundingBox<itk::PointSet<float,3>::PointIdentifier,3,float,itk::PointSet<float,3>::PointsContainer> BBox;
  BBox::Pointer boundingBox = BBox::New();
  boundingBox->SetPoints(movingMesh->GetPoints());
  boundingBox->ComputeBoundingBox();
  itk::Point<float> center = boundingBox->GetCenter();
  std::cout << "Mesh center = " << center << std::endl;
  PointsToImageFilterType::Pointer pointsToImageFilter = PointsToImageFilterType::New();
  pointsToImageFilter->SetInput( movingMesh );

  BinaryImageType::SpacingType spacing;
  spacing.Fill( 1.0 );
  BinaryImageType::PointType origin;
  origin = center;
  pointsToImageFilter->SetSpacing( spacing );
  pointsToImageFilter->SetOrigin( origin   );
  pointsToImageFilter->Update();

  DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
  distanceFilter->SetInput( pointsToImageFilter->GetOutput() );
  distanceFilter->Update();
  registration->SetMovingImage(distanceFilter->GetOutput());

  // Connect an observer
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  metric->SetFixedPointSet(fixedMesh);
  metric->SetMovingImage(distanceFilter->GetOutput());

//   itk::ImageFileWriter<BinaryImageType>::Pointer w1 = itk::ImageFileWriter<BinaryImageType>::New();
//   w1->SetFileName("./binaryImage.mha");
//   w1->SetInput(pointsToImageFilter->GetOutput());
//   w1->Update();
//   itk::ImageFileWriter<DistanceImageType>::Pointer w2 = itk::ImageFileWriter<DistanceImageType>::New();
//   w2->SetFileName("./distanceMap.mha");
//   w2->SetInput(distanceFilter->GetOutput());
//   w2->Update();
  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Execute registration..." << std::endl;
  }
  try
  {
    registration->Update();
    std::cout << "Optimizer stop condition: "
    << registration->GetOptimizer()->GetStopConditionDescription()
    << std::endl;
  }
  catch( itk::ExceptionObject & e )
  {
    std::cout << e << std::endl;
  }

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "registerMeshes(): Finished registration..." << std::endl;
  }
  return transform;
}

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  PARSE_ARGS;

  typedef float PixelType;
  typedef itk::Mesh<PixelType,3> MeshType;
  typedef itk::VTKTetrahedralMeshReader<MeshType> MeshReaderType;

  typedef itk::TranslationTransform<double,3> TransformType;

  if (Verbose)
  {
    std::cout << "Read data..." << std::endl;
  }

  // Read vtk data
  MeshReaderType::Pointer vtkFixedMeshReader = MeshReaderType::New();
  vtkFixedMeshReader->SetFileName(FixedImageMesh);

  MeshReaderType::Pointer vtkMovingMeshReader = MeshReaderType::New();
  vtkMovingMeshReader->SetFileName(MovingImageMesh);

  try
  {
    std::cerr << "Trying to read..." << std::endl;
    vtkFixedMeshReader->Update();
    vtkMovingMeshReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Error during vtkFixedMeshReader->Update() and vtkMovingMeshReader->Update() " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  if (Verbose)
  {
    std::cout << "Done readinga..." << std::endl;
  }
  MeshType::Pointer fixedMesh = vtkFixedMeshReader->GetOutput();
  MeshType::Pointer movingMesh = vtkMovingMeshReader->GetOutput();

  if (Verbose)
  {
    std::cout << "Total # fixed tetras..." << fixedMesh->GetNumberOfCells() << std::endl;
    std::cout << "Total # moving tetras..." << movingMesh->GetNumberOfCells() << std::endl;
  }


  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Register meshes using ICP..." << std::endl;
  }

  TransformType::Pointer transform =
  registerMeshes(fixedMesh.GetPointer(),movingMesh.GetPointer(),
                 Iterations,GradientTolerance,ValueTolerance,
                 epsilonFunction,Verbose);

  itk::TransformFileWriter::Pointer writer = itk::TransformFileWriter::New();
  writer->SetInput(transform);
  writer->SetFileName(OutputTransform);

  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cout << e << std::endl;
  }

  return EXIT_SUCCESS;
}

