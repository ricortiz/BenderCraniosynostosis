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

// Bender includes
#include "FEMAnalysisCLP.h"
#include "benderIOUtils.h"
#include "itkVTKTetrahedralMeshReader.h"

// OpenGL includes
//#include <GL/glew.h>
//#include <GL/glut.h>

// SOFA includes
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/mapping/BarycentricMappingRigid.h>
#include <sofa/component/misc/RequiredPlugin.h>
#include <sofa/component/misc/VTKExporter.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/typedef/Sofa_typedef.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/vector.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/graph/DAGSimulation.h>

// SofaFlexible includes
#include <plugins/Flexible/quadrature/TopologyGaussPointSampler.h>
#include <plugins/Flexible/shapeFunction/BarycentricShapeFunction.h>
#include <plugins/Flexible/deformationMapping/LinearMapping.h>
#include <plugins/Flexible/strainMapping/PrincipalStretchesMapping.h>
#include <plugins/Flexible/material/StabilizedNeoHookeanForceField.h>

// SofaCUDA includes
#ifdef SOFA_CUDA
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaTetrahedronFEMForceField.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaCollisionDetection.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaMechanicalObject.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaTriangleObject.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaLineModel.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaPointModel.h>
#include <plugins/SofaCUDA/sofa/gpu/cuda/CudaUniformMass.h>
#endif

// ITK include
#include <itkImageFileReader.h>
#include <itkImage.h>
#include <itkMesh.h>
#include <itkTetrahedronCell.h>
#include <itkVector.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkTranslationTransform.h>
#include <itkEuclideanDistancePointMetric.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <itkPointSetToPointSetRegistrationMethod.h>
#include <itkDanielssonDistanceMapImageFilter.h>
#include <itkPointSetToImageFilter.h>

// BenderVTK includes
#include <vtkDualQuaternion.h>

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
using namespace sofa::component::container;
using namespace sofa::component::engine;
using namespace sofa::component::forcefield;
using namespace sofa::component::linearsolver;
using namespace sofa::component::mapping;
using namespace sofa::component::odesolver;
using namespace sofa::component::projectiveconstraintset;
using namespace sofa::component::topology;
using namespace sofa::component::visualmodel;
using namespace sofa::component::shapefunction;
using namespace sofa::helper;
using namespace sofa::simulation;

class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {};
  static int iteration;
public:

  typedef itk::LevenbergMarquardtOptimizer     OptimizerType;
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

//     std::cout << "Value = " << optimizer->GetCachedValue() << std::endl;
    std::cout << "Position = "  << optimizer->GetCachedCurrentPosition() << std::endl;
    std::cout << "Iteration = " << ++iteration << std::endl;
    std::cout << std::endl << std::endl;

  }

};

int CommandIterationUpdate::iteration = 0;
/// helper function for more compact component creation
// ---------------------------------------------------------------------
template<class Component>
typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="" )
{
  typename Component::SPtr component =
    sofa::core::objectmodel::New<Component>();
  parentNode->addObject(component);
  component->setName(parentNode->getName()+"_"+name);
  return component;
}

// Copy point positions from vtk to a mechanical object
// ---------------------------------------------------------------------
void copyVertices( itk::Mesh<float,3>::PointsContainer * points,
                   MechanicalObject<Vec3Types>* mechanicalMesh)
{
  vtkIdType numberOfPoints = points->Size();

  vtkIdType meshPointId = mechanicalMesh->getSize() > 1 ? mechanicalMesh->getSize() : 0;
  mechanicalMesh->resize(numberOfPoints);

  Data<MechanicalObject<Vec3Types>::VecCoord>* x =
    mechanicalMesh->write(sofa::core::VecCoordId::position());

  // Copy vertices from vtk mesh
  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();

  for(vtkIdType i = 0; i < numberOfPoints; ++i)
    {
    itk::Point <float> vertex = points->GetElement(i);
    Vector3 point;
    point[0] = vertex[0];
    point[1] = vertex[1];
    point[2] = vertex[2];
    vertices[meshPointId++] = point;
    }
  if (meshPointId != numberOfPoints)
    {
    std::cerr << "Failed to copy vertices: " << numberOfPoints << " vs "
              << meshPointId << std::endl;
    }
  x->endEdit();
}

/// Visualization node (for debug purposes only)
// ---------------------------------------------------------------------
Node::SPtr createVisualNode(Node *                        parentNode,
                            vtkPolyData *                 polyMesh,
                            MechanicalObject<Vec3Types> * mechanicalObject,
                            int                           label = 0
                            )
{

  vtkNew<vtkDataSetSurfaceFilter> surfaceExtractor;

  if(label != 0)
    {
    vtkNew<vtkThreshold> meshThreshold;
    meshThreshold->SetInput(polyMesh);
    meshThreshold->ThresholdBetween(label,label);
    surfaceExtractor->SetInput(meshThreshold->GetOutput());
    }
  else
    {
    surfaceExtractor->SetInput(polyMesh);

    }
  surfaceExtractor->Update();

  Node::SPtr     visualNode = parentNode->createChild("visualNode");
  OglModel::SPtr oglModel   = addNew<OglModel>(visualNode,"oglModel");

  vtkNew<vtkPolyDataNormals> surfaceNormals;
  surfaceNormals->SetInput(surfaceExtractor->GetOutput());
  surfaceNormals->ComputeCellNormalsOn();
  surfaceNormals->Update();

  vtkFloatArray *cellNormals = vtkFloatArray::SafeDownCast(
    surfaceNormals->GetOutput()->GetCellData()->GetNormals());

  ResizableExtVector<Vec3f> normals;
  normals.reserve(cellNormals->GetNumberOfTuples());

  for(vtkIdType i = 0, end = cellNormals->GetNumberOfTuples(); i < end; ++i)
    {
    Vec3f normal;
    cellNormals->GetTupleValue(i,normal.ptr());
    normals.push_back(normal);
    }
  oglModel->setVnormals(&normals);

  IdentityMapping<Vec3Types,
                  ExtVec3fTypes>::SPtr identityMapping =
    addNew<IdentityMapping<Vec3Types, ExtVec3fTypes> >(visualNode,
      "identityMapping");
  identityMapping->setModels(mechanicalObject,oglModel.get());

  return visualNode;
}

// ---------------------------------------------------------------------
MechanicalObject<Vec3Types>::SPtr createGhostPoints(
  Node *       parentNode,
  itk::TranslationTransform< double,3>* transform,
  itk::Mesh<float,3> *mesh
  )
{
  typedef itk::Mesh<float,3>::PointsContainer PointsContainer;
  itk::TranslationTransform< double,3>::OutputPointType OutputPointType;

  MechanicalObject<Vec3Types>::SPtr ghostDOF =
    addNew<MechanicalObject<Vec3Types> >(parentNode, "ghostDOF");

  // Get positions
  size_t numberOfPoints = mesh->GetNumberOfPoints();
  std::cout << "Number of points: " << numberOfPoints << std::endl;

  // Traverse mesh nodes
  PointsContainer *points = mesh->GetPoints();
  itk::TranslationTransform< double,3>::InputPointType  fixedPoint;
  itk::TranslationTransform< double,3>::OutputPointType movingPoint;

  ghostDOF->resize(numberOfPoints);
  Data<MechanicalObject<Vec3Types>::VecCoord> *x =
    ghostDOF->write(sofa::core::VecCoordId::position());

  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();
  for(size_t i = 0; i < numberOfPoints; ++i)
    {
    fixedPoint = points->GetElement(i);
  movingPoint = transform->TransformPoint(fixedPoint);

    vertices[i][0] = fixedPoint[0] + movingPoint[0];
    vertices[i][1] = fixedPoint[1] + movingPoint[1];
    vertices[i][2] = fixedPoint[2] + movingPoint[2];

    }
  x->endEdit();

  UniformMass3::SPtr frameMass = addNew<UniformMass3>(parentNode,"FrameMass");
  frameMass->setTotalMass(1);

  return ghostDOF;
}


/// Create a FEM in parentNode.  A MeshTopology should be defined in
/// parentNode prior to calling this function.
// ---------------------------------------------------------------------
void createFiniteElementModel(Node* parentNode, bool linearFEM,
                              const Vec3Types::Real &youngModulus,
                              const Vec3Types::Real &poissonRatio  )
{
  if (linearFEM)
    {
    TetrahedronFEMForceField< Vec3Types >::SPtr femSolver =
      addNew<TetrahedronFEMForceField< Vec3Types > >(parentNode,"femSolver");
    femSolver->setComputeGlobalMatrix(false);
    femSolver->setMethod("large");
    femSolver->setPoissonRatio(.35);
    femSolver->setYoungModulus(youngModulus);
    return;
    }

  Node::SPtr behavior = parentNode->createChild("behavior");
  MechanicalObject<Vec3Types> *tetMesh =
    dynamic_cast<MechanicalObject<Vec3Types>*>( parentNode->getMechanicalState());

  // Define control nodes
  BarycentricShapeFunction<ShapeFunction3>::SPtr shapeFunction =
    addNew<BarycentricShapeFunction<ShapeFunction3> >(parentNode,"shapeFunction");

  // Sample mesh where the deformation gradients re going to be defined
  TopologyGaussPointSampler::SPtr sampler = addNew<TopologyGaussPointSampler>(behavior,"sampler");
  sampler->f_inPosition.setParent(&tetMesh->x);

  MechanicalObject<F331Types>::SPtr F = addNew<MechanicalObject<F331Types> >(behavior,"F");

  // Map mesh to sampled nodes
  LinearMapping<Vec3Types,F331Types>::SPtr linearMapping =
    addNew<LinearMapping<Vec3Types,F331Types> >(behavior,"linearMapping");
  linearMapping->setModels(tetMesh,F.get());

  // Create strain measurements
  Node::SPtr strainNode = behavior->createChild("strain");
    {
    MechanicalObject<U331Types>::SPtr U = addNew<MechanicalObject<U331Types> >(strainNode,"U");
    PrincipalStretchesMapping<F331Types,U331Types>::SPtr principalMapping =
      addNew<PrincipalStretchesMapping<F331Types,U331Types> >(strainNode,"principalMapping");
    principalMapping->threshold.setValue(0.6);
    principalMapping->asStrain.setValue(false);
    principalMapping->setModels(F.get(),U.get());

    StabilizedNeoHookeanForceField<U331Types>::SPtr forceField =
      addNew<StabilizedNeoHookeanForceField<U331Types> >(strainNode,"Force Field");

    Vec3Types::VecReal &modulus = *forceField->_youngModulus.beginEdit();
    modulus[0] = youngModulus;
    forceField->_youngModulus.endEdit();
    Vec3Types::VecReal &poisson = *forceField->_poissonRatio.beginEdit();
    poisson[0] = poissonRatio;
    forceField->_poissonRatio.endEdit();
    }
}

/// Loads a vtk tetrahedral polymesh and creates a mechanical object and
/// the corresponding MeshTopology.
// ---------------------------------------------------------------------
MechanicalObject<Vec3Types>::SPtr loadMesh(Node*               parentNode,
                                           itk::Mesh<float,3> *polyMesh
                                           )
{
  typedef itk::Mesh<float,3>::CellType CellType;
  typedef itk::Mesh<float,3>::CellsContainer::ConstIterator  CellIterator;
  typedef itk::TetrahedronCell< CellType >  TetrahedronType;

  // load mesh
  itk::Mesh<float,3>::PointsContainer *points = polyMesh->GetPoints();
  itk::Mesh<float,3>::CellsContainer *tetras = polyMesh->GetCells();

  std::stringstream meshName;
  meshName << "Mesh";

  // Create mechanical object (dof) for the mesh and extract material parameters
  MechanicalObject<Vec3Types>::SPtr mechanicalMesh =
    addNew<MechanicalObject<Vec3Types> >(parentNode,meshName.str());

  copyVertices(points,mechanicalMesh.get());

  // Create the MeshTopology
  MeshTopology::SPtr meshTopology = addNew<MeshTopology>(parentNode, "Topology");
  meshTopology->seqPoints.setParent(&mechanicalMesh->x);

  // Copy tetrahedra array from vtk cell array
  MeshTopology::SeqTetrahedra& tetrahedra =
    *meshTopology->seqTetrahedra.beginEdit();

  tetrahedra.reserve(polyMesh->GetNumberOfCells());

  std::cout << "Total # of tetrahedra: " << polyMesh->GetNumberOfCells()
            << std::endl;

 CellIterator cellIterator = tetras->Begin();
 CellIterator cellEnd = tetras->End();
 while(cellIterator != cellEnd)
 {
   CellType *cell = cellIterator.Value();
   TetrahedronType *tet = static_cast<TetrahedronType*>(cell);
   const itk::IdentifierType *vertexIds = tet->GetPointIds();
   tetrahedra.push_back(MeshTopology::Tetra(vertexIds[0], vertexIds[1], vertexIds[2], vertexIds[3]));
   ++cellIterator;
 }
 meshTopology->seqTetrahedra.endEdit();
 return mechanicalMesh;

}


void createEulerSolverNode(Node::SPtr         parentNode,
                           const std::string &scheme = "Explicit")
{
  typedef EulerImplicitSolver EulerImplicitSolverType;
  typedef EulerSolver EulerExplicitSolverType;
  typedef CGLinearSolver<GraphScatteredMatrix,
                         GraphScatteredVector> CGLinearSolverType;

  // Implicit time-step method requires a linear solver
  if (scheme == "Implicit")
    {
    EulerImplicitSolverType::SPtr odeSolver =
      addNew<EulerImplicitSolverType>(parentNode,"TimeIntegrator");

    CGLinearSolverType::SPtr linearSolver = addNew<CGLinearSolverType>(
      parentNode,"CGSolver");
    odeSolver->f_rayleighStiffness.setValue(0.01);
    odeSolver->f_rayleighMass.setValue(1);

    linearSolver->f_maxIter.setValue(25);                 //iteration maxi for the CG
    linearSolver->f_smallDenominatorThreshold.setValue(1e-05);
    linearSolver->f_tolerance.setValue(1e-05);
    }
  else if (scheme == "Explicit")
    {
    EulerExplicitSolverType::SPtr solver = addNew<EulerExplicitSolverType>(
      parentNode,"TimeIntegrator");
    }
  else
    {
    std::cerr << "Error: " << scheme <<
      " Integration Scheme not recognized" <<
      std::endl;
    }
}

//------------------------------------------------------------------------------
void initMesh(vtkPolyData* outputPolyData, Node::SPtr anatomicalMesh)
{
  MeshTopology *topology = anatomicalMesh->getNodeObject<MeshTopology>();
  MechanicalObject3 *dof = anatomicalMesh->getNodeObject<MechanicalObject3>();

  vtkNew<vtkPoints> points;
  const vtkIdType numberOfPoints = topology->getNbPoints();
  points->SetNumberOfPoints(numberOfPoints);
  for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
    {
    points->InsertPoint(pointId, topology->getPX(pointId),
                        topology->getPY(pointId),
                        topology->getPZ(pointId));
    }
  outputPolyData->SetPoints(points.GetPointer());
  // Cells
  vtkNew<vtkCellArray> cells;
  for (vtkIdType cellId = 0; cellId < topology->getNbTetras(); ++cellId)
    {
    const Tetra& t = topology->getTetra(cellId);
    vtkIdType cell[4];
    cell[0] = t[0];
    cell[1] = t[1];
    cell[2] = t[2];
    cell[3] = t[3];
    cells->InsertNextCell(4, cell);
    }
  outputPolyData->SetPolys(cells.GetPointer());

  vtkNew<vtkFloatArray> forceVectors;
  forceVectors->SetNumberOfComponents(3);

  const VecDeriv3 &forces = *dof->getF();
  forceVectors->SetNumberOfTuples(forces.size());

  for(size_t i = 0, end = forces.size(); i< end; ++i)
  {
    float f[3] = {0};
    f[0] = forces[i][0];
    f[1] = forces[i][1];
    f[2] = forces[i][2];

    forceVectors->SetTupleValue(i,f);
  }

  outputPolyData->GetPointData()->AddArray(forceVectors.GetPointer());

}

//------------------------------------------------------------------------------
double meanSquareError(MechanicalObject<Vec3Types>::SPtr mesh1,
                       MechanicalObject<Vec3Types>::SPtr mesh2)
{
  const Data<MechanicalObject<Vec3Types>::VecCoord>* position1 =
    mesh1->read(VecCoordId::position());
  const Data<MechanicalObject<Vec3Types>::VecCoord>* position2 =
    mesh2->read(VecCoordId::position());

  if (!position1 || !position2)
    {
    std::cerr << "No positions: " << position1 << ", " << position2 << std::endl;
    return -1.;
    }
  // Copy vertices from vtk mesh
  const MechanicalObject<Vec3Types>::VecCoord& vertices1 = position1->getValue();
  const MechanicalObject<Vec3Types>::VecCoord& vertices2 = position2->getValue();

  size_t numberOfPoints = vertices1.size();
  if (numberOfPoints != vertices2.size())
    {
    std::cerr << "Not the same number of vertices: "
              << vertices1.size() << " != " << vertices2.size() << std::endl;
    return -1.;
    }

  double error = 0.;
  MechanicalObject<Vec3Types>::VecCoord::const_iterator it1;
  MechanicalObject<Vec3Types>::VecCoord::const_iterator it2;
  for(it1 = vertices1.begin(), it2 = vertices2.begin();
      it1 != vertices1.end();
      ++it1, ++it2)
    {
    Vector3 distance = *it1 - *it2;
    error += distance.norm2() / numberOfPoints;
    }
  return error;
}

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
  PointsToImageFilterType::Pointer pointsToImageFilter = PointsToImageFilterType::New();
  pointsToImageFilter->SetInput( fixedMesh );
  BinaryImageType::SpacingType spacing;
  spacing.Fill( 1.0 );
  BinaryImageType::PointType origin;
  origin.Fill( 0.0 );
  pointsToImageFilter->SetSpacing( spacing );
  pointsToImageFilter->SetOrigin( origin   );
  pointsToImageFilter->Update();
  BinaryImageType::Pointer binaryImage = pointsToImageFilter->GetOutput();
  typedef itk::Image< unsigned short, 3 >  DistanceImageType;
  typedef itk::DanielssonDistanceMapImageFilter<BinaryImageType, DistanceImageType> DistanceFilterType;
  DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
  distanceFilter->SetInput( binaryImage );
  distanceFilter->Update();
  metric->SetDistanceMap( distanceFilter->GetOutput() );
  // Connect an observer
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );
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

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  PARSE_ARGS;

  typedef float PixelType;
  typedef itk::Mesh<PixelType,3> MeshType;

  typedef itk::VTKTetrahedralMeshReader<MeshType> MeshReaderType;

  const double dt = 0.0001;

  sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

  // Create the scene graph root node
  Node::SPtr root = getSimulation()->createNewGraph("root");
  root->setGravity( Coord3(0,0,0) );
  root->setDt(dt);

#ifdef SOFA_CUDA
  // Load SofaCUDA plugin
  sofa::component::misc::RequiredPlugin::SPtr cudaPlugin =
    addNew<sofa::component::misc::RequiredPlugin>(root,"CUDA");
  cudaPlugin->pluginName.setValue("SofaCUDA");
#endif

  if (!IsMeshInRAS)
    {
    std::cout<<"Mesh x,y coordinates will be inverted" << std::endl;
    }

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

  MeshType::Pointer fixedMesh = vtkFixedMeshReader->GetOutput();
  MeshType::Pointer movingMesh = vtkMovingMeshReader->GetOutput();

  if (Verbose)
  {
    std::cout << "************************************************************"
    << std::endl;
    std::cout << "Register meshes using ICP..." << std::endl;
  }
  itk::TranslationTransform< double,3>::Pointer transform =
  registerMeshes(fixedMesh.GetPointer(),movingMesh.GetPointer(),
                 Iterations,GradientTolerance,ValueTolerance,
                 epsilonFunction,Verbose);

  // Create a scene node
  Node::SPtr sceneNode = root->createChild("FEMSimulation");

  // Time stepper for the armature
  createEulerSolverNode(root.get(),"Implicit");

  Node::SPtr ghostDOFNode = root->createChild("BoundaryCondition");

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create ghost mesh..." << std::endl;
    }

  Vector6 box;
  MechanicalObject<Vec3Types>::SPtr ghostDOF =
  createGhostPoints(ghostDOFNode.get(), transform, movingMesh );

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create Skull mesh..." << std::endl;
    }

  // Node for the mesh
  Node::SPtr skullDOFNode = sceneNode->createChild("SkullMesh");

  // Create mesh dof
  MechanicalObject<Vec3Types>::SPtr skullDOF = loadMesh(skullDOFNode.get(), fixedMesh);
  UniformMass3::SPtr anatomicalMass = addNew<UniformMass3>(skullDOFNode.get(),"Mass");
  anatomicalMass->setTotalMass(100);

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create finite element model..." << std::endl;
    }

  // Finite element method
  createFiniteElementModel(skullDOFNode.get(), LinearFEM, youngModulus, poissonRatio);

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create force loads..." << std::endl;
    }
  using sofa::component::interactionforcefield::StiffSpringForceField;

  StiffSpringForceField<Vec3Types>::SPtr stiffspringforcefield =
  sofa::core::objectmodel::New<StiffSpringForceField<Vec3Types> >(ghostDOF.get(),skullDOF.get());
  stiffspringforcefield->setName("Force Loads");
  skullDOFNode->addObject(stiffspringforcefield);

  double stiffness = 10000.;
  double distance = 1.;
  const vtkIdType numberOfPoints = fixedMesh->GetPoints()->Size();
  size_t sample = 0;
  for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
    {
    if (!(sample++ % 1))
      {
      stiffspringforcefield->addSpring(pointId, pointId, stiffness, 0.0, distance);
      }
    }

  const sofa::core::objectmodel::TagSet &tags = skullDOF->getTags();
  for (sofa::core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
    stiffspringforcefield->addTag(*it);

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    }
  // Run simulation time steps
  if (Debug)
    {
    std::string sceneFileName = OutputTetMesh;
    sceneFileName += ".scn";
    std::cout << "Write scene at " << sceneFileName << std::endl;
    sofa::simulation::getSimulation()->exportXML(root.get(), sceneFileName.c_str());
    }
  if (Verbose)
    {
    std::cout << "Init..." << std::endl;
    }
  sofa::simulation::getSimulation()->init(root.get());

  int gluArgc  = 1;
  char** gluArgv = new char *;
  gluArgv[0] = new char[strlen(argv[0])+1];
  memcpy(gluArgv[0], argv[0], strlen(argv[0])+1);
  glutInit(&gluArgc, gluArgv);

  if (GUI)
    {
    std::cout << "Open GUI..." << std::endl;
    //
    sofa::gui::initMain();
    sofa::gui::GUIManager::Init(gluArgv[0]);
    //root->setAnimate(true);
    int err = sofa::gui::GUIManager::MainLoop(root);
    if (err)
      {
      std::cerr << "Error in SOFA. " << std::endl;
      return err;
      }
    }
  else
    {
    if (Verbose)
      {
      std::cout << "Create OpenGL context..." << std::endl;
      }
    glutCreateWindow(argv[0]);
  //glewExperimental=true;

    root->setAnimate(true);

    if (Verbose)
      {
      std::cout << "Animate..." << std::endl;
      }

    // Forces take time to start moving the mesh
    const size_t minimumNumberOfSteps = 30;

    double lastError = 1.;
    double stdDeviation = 0.;

    // We can't use the distance error directly because the simulation might
    // oscillate.
    for (size_t step = 0;
         (step < minimumNumberOfSteps || stdDeviation > MinimumStandardDeviation) &&
         (step < static_cast<size_t>(MaximumNumberOfSimulationSteps)) ; ++step)
      {
      sofa::simulation::getSimulation()->animate(root.get(), dt);
      //sofa::simulation::getSimulation()->animate(root.get());

      const double error = meanSquareError(ghostDOF, skullDOF);
      double mean = (lastError + error) / 2.;
      stdDeviation = sqrt((pow(lastError - mean, 2) + pow(error - mean, 2)) / 2.);
      //errorChange =  fabs(lastError-error) / lastError;
      lastError = error;

      if (Verbose)
        {
        std::cout << " Iteration #" << step << " (distance: " << lastError
                    << " std: " << stdDeviation << std::endl;
        }
      }
    }
  vtkNew<vtkPolyData> posedSurface;
  initMesh(posedSurface.GetPointer(), skullDOFNode);
  if (!IsMeshInRAS)
    {
    vtkNew<vtkTransform> transform;
    transform->RotateZ(180.0);

    vtkNew<vtkTransformPolyDataFilter> transformer;
    transformer->SetInput(posedSurface.GetPointer());
    transformer->SetTransform(transform.GetPointer());
    transformer->Update();

    bender::IOUtils::WritePolyData(transformer->GetOutput(), OutputTetMesh);
    }
  else
    {
    bender::IOUtils::WritePolyData(posedSurface.GetPointer(), OutputTetMesh);
    }

  if (Verbose)
    {
    std::cout << "Unload..." << std::endl;
    }
  sofa::simulation::getSimulation()->unload(root);

  return EXIT_SUCCESS;
}

