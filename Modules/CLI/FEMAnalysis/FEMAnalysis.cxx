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
#include "SimulatePoseCLP.h"
#include "benderIOUtils.h"
#include "vtkQuaternion.h"

// OpenGL includes
//#include <GL/glew.h>
//#include <GL/glut.h>

// SOFA includes
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/LocalMinDistance.h>
#include <sofa/component/collision/MinProximityIntersection.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/engine/BoxROI.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/mapping/BarycentricMappingRigid.h>
#include <sofa/component/misc/RequiredPlugin.h>
#include <sofa/component/misc/VTKExporter.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>
#include <sofa/component/projectiveconstraintset/SkeletalMotionConstraint.h>
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
#include <itkImage.h>
#include <itkVector.h>
#include <itkImageFileReader.h>
#include <itkVectorLinearInterpolateImageFunction.h>

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
using namespace sofa::component::collision;
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
void copyVertices( vtkPoints* points,
                   MechanicalObject<Vec3Types>* mechanicalMesh)
{
  vtkIdType numberOfPoints = points->GetNumberOfPoints();

  vtkIdType meshPointId = mechanicalMesh->getSize() > 1 ? mechanicalMesh->getSize() : 0;
  mechanicalMesh->resize(numberOfPoints);

  Data<MechanicalObject<Vec3Types>::VecCoord>* x =
    mechanicalMesh->write(sofa::core::VecCoordId::position());

  // Copy vertices from vtk mesh
  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();

  for(vtkIdType i = 0, end = points->GetNumberOfPoints(); i < end; ++i)
    {
    Vector3 point;
    points->GetPoint(i,point.ptr());
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
  itk::Image<itk::Vector<float,3>,3>* displacementField,
  vtkPolyData* mesh
  )
{
  MechanicalObject<Vec3Types>::SPtr ghostDOF =
    addNew<MechanicalObject<Vec3Types> >(parentNode, "ghostDOF");

  // Get positions
  size_t numberOfPoints = mesh->GetNumberOfPoints();
  std::cout << "Number of points: " << numberOfPoints << std::endl;

  // Traverse mesh nodes
  vtkSmartPointer<vtkPoints> points = mesh->GetPoints();
  itk::VectorLinearInterpolateImageFunction<itk::Image<itk::Vector<float,3>,3>, double>::Pointer interpolator
    = itk::VectorLinearInterpolateImageFunction<itk::Image<itk::Vector<float,3>,3>, double>::New();
  interpolator->SetInputImage(displacementField);

  ghostDOF->resize(numberOfPoints);
  Data<MechanicalObject<Vec3Types>::VecCoord> *x =
    ghostDOF->write(sofa::core::VecCoordId::position());

  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();
  for(size_t i = 0; i < numberOfPoints; ++i)
    {
    Vector3 point;
    points->GetPoint(i, point.ptr());

    itk::Point<double> index;
    index[0] = point[0];
    index[1] = point[1];
    index[2] = point[2];
    itk::Vector<float,3> dx = interpolator->Evaluate(index);

    vertices[i][0] = point[0]+dx[0];
    vertices[i][1] = point[1]+dx[1];
    vertices[i][2] = point[2]+dx[2];

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
                                           vtkPolyData *       polyMesh
                                           )
{
  // load mesh
  vtkSmartPointer<vtkPoints>    points;
  vtkSmartPointer<vtkCellArray> tetras;
  vtkSmartPointer<vtkCellData>  data;

  points = polyMesh->GetPoints();
  tetras = polyMesh->GetPolys();
  data   = polyMesh->GetCellData();

  std::stringstream meshName;
  meshName << "Mesh";

  // Create mechanical object (dof) for the mesh and extract material parameters
  MechanicalObject<Vec3Types>::SPtr mechanicalMesh =
    addNew<MechanicalObject<Vec3Types> >(parentNode,meshName.str());

  copyVertices(points.GetPointer(),mechanicalMesh.get());

  // Create the MeshTopology
  MeshTopology::SPtr meshTopology = addNew<MeshTopology>(parentNode, "Topology");
  meshTopology->seqPoints.setParent(&mechanicalMesh->x);

  // Copy tetrahedra array from vtk cell array
  MeshTopology::SeqTetrahedra& tetrahedra =
    *meshTopology->seqTetrahedra.beginEdit();
  tetrahedra.reserve(tetras->GetNumberOfCells());

  std::cout << "Total # of tetrahedra: " << tetras->GetNumberOfCells()
            << std::endl;

  tetras->InitTraversal();

  vtkNew<vtkIdList> element;
  for (vtkIdType cellId = 0; tetras->GetNextCell(element.GetPointer());++cellId)
    {
    if(element->GetNumberOfIds() != 4)
      {
      std::cerr << "Error: Non-tetrahedron encountered." << std::endl;
      continue;
      }

    tetrahedra.push_back(MeshTopology::Tetra(element->GetId(0), element->GetId(1), element->GetId(2), element->GetId(3)));
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
void initMesh(vtkPolyData* outputPolyData, vtkPolyData* inputPolyData,
              Node::SPtr anatomicalMesh)
{
  MeshTopology *topology = anatomicalMesh->getNodeObject<MeshTopology>();
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

  for (int i = 0; i < inputPolyData->GetPointData()->GetNumberOfArrays(); ++i)
    {
    outputPolyData->GetPointData()->AddArray(inputPolyData->GetPointData()->GetArray(i));
    }
  for (int i = 0; i < inputPolyData->GetCellData()->GetNumberOfArrays(); ++i)
    {
    outputPolyData->GetCellData()->AddArray(inputPolyData->GetCellData()->GetArray(i));
    }

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

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  PARSE_ARGS;

  typedef itk::Vector<float,3> ImagePixelType;
  typedef itk::Image<ImagePixelType,3> DisplacementFieldType;
  typedef itk::ImageReader<DisplacementFieldType> DisplacementFieldReaderType;

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
  vtkSmartPointer<vtkPolyData> fixedMesh;
  fixedMesh.TakeReference(
    bender::IOUtils::ReadPolyData(FixedImageMesh.c_str(),!IsMeshInRAS));

  vtkSmartPointer<vtkPolyData> movingMesh;
  movingMesh.TakeReference(
    bender::IOUtils::ReadPolyData(MovingImageMesh.c_str(),!IsMeshInRAS));

  DisplacementFieldType::Pointer displacementField;
  DisplacementFieldReaderType::Pointer reader = DisplacementFieldReaderType::New();
  reader->SetFileName(DisplacementField.c_str());
  reader->Update();
  displacementField = reader->GetOutput();

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

  // In GUI mode, there is no animation decomposition, forces are computed
  // between the final pose and the start pose. In non-GUI mode, forces are
  // recomputed at each step.
  const double firstFrame = (GUI ? 1. : 1. / NumberOfArmatureSteps);
  Vector6 box;
  MechanicalObject<Vec3Types>::SPtr ghostDOF =
  createGhostPoints(ghostDOFNode.get(), displacementField, movingMesh );

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
    std::cout << "Create spring forces..." << std::endl;
    }
  using sofa::component::interactionforcefield::StiffSpringForceField;

  StiffSpringForceField<Vec3Types>::SPtr stiffspringforcefield =
  sofa::core::objectmodel::New<StiffSpringForceField<Vec3Types> >(ghostDOF.get(),skullDOF.get());
  stiffspringforcefield->setName("Force Loads");
  skullDOFNode->addObject(stiffspringforcefield);

  double stiffness = 10000.;
  double distance = 1.;
  const vtkIdType numberOfPoints = fixedMesh->GetPoints()->GetNumberOfPoints();
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
  initMesh(posedSurface.GetPointer(), fixedMesh, skullDOFNode);
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

