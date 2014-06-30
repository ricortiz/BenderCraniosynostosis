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
#include "itkVTKPointCloudReader.h"

// Eigen includes
#include <Eigen/Eigenvalues>

// OpenGL includes
//#include <GL/glew.h>
//#include <GL/glut.h>

// SOFA includes
#include <sofa/component/linearsolver/CGLinearSolver.h>
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
#include <sofa/component/topology/Mesh2PointTopologicalMapping.h>
#include <sofa/component/topology/PointSetTopologyContainer.h>

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
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>
#include <itkEuclideanDistancePointMetric.h>
#include <itkLevenbergMarquardtOptimizer.h>
#include <itkPointSetToPointSetRegistrationMethod.h>
#include <itkDanielssonDistanceMapImageFilter.h>
#include <itkPointSetToImageFilter.h>
#include <itkPointsLocator.h>
#include <itkResampleImageFilter.h>
#include <itkBSplineScatteredDataPointSetToImageFilter.h>
#include <itkNearestNeighborExtrapolateImageFunction.h>

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
using namespace sofa::component::interactionforcefield;

class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate Self;
  typedef  itk::Command Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );

protected:
  CommandIterationUpdate() {
  };
  static int iteration;
public:

  typedef itk::LevenbergMarquardtOptimizer OptimizerType;
  typedef const OptimizerType *                OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    OptimizerPointer optimizer =
      dynamic_cast< OptimizerPointer >( object );

    if( !itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }
    std::cout << "Position = "  << optimizer->GetCachedCurrentPosition() << std::endl;
    std::cout << "Iteration = " << ++iteration << std::endl;

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
void copyVertices( itk::PointSet<float,3>::PointsContainer * points,
                   MechanicalObject<Vec3Types>* mechanicalMesh)
{
  vtkIdType numberOfPoints = points->Size();

  vtkIdType meshPointId = mechanicalMesh->getSize() >
                          1 ? mechanicalMesh->getSize() : 0;
  mechanicalMesh->resize(numberOfPoints);

  Data<MechanicalObject<Vec3Types>::VecCoord>* x =
    mechanicalMesh->write(sofa::core::VecCoordId::position());

  // Copy vertices from vtk mesh
  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();

  for(vtkIdType i = 0; i < numberOfPoints; ++i)
    {
    itk::Point <float> vertex = points->GetElement(i);
    Vector3            point;
    point[0]                = vertex[0];
    point[1]                = vertex[1];
    point[2]                = vertex[2];
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
MechanicalObject<Vec3Types>::SPtr loadFixedPointCloud(
  Node * parentNode,
  itk::PointSet<float,3> *pointCloud,
  bool Verbose
  )
{
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create fixed point mechanical object..." << std::endl;
    }
  typedef itk::PointSet<float,3>::PointsContainer PointsContainer;

  MechanicalObject<Vec3Types>::SPtr fixedCloudDOF =
    addNew<MechanicalObject<Vec3Types> >(parentNode, "fixedCloudDOF");

  // Get positions
  size_t numberOfPoints = pointCloud->GetNumberOfPoints();
  std::cout << "Number of points: " << numberOfPoints << std::endl;

  // Traverse pointCloud nodes
  PointsContainer *points = pointCloud->GetPoints();

  copyVertices(points,fixedCloudDOF.get());

  // Create the Point Set Topology
  PointSetTopologyContainer::SPtr pointSetTopology =
  addNew<PointSetTopologyContainer>(parentNode, "Topology");
  pointSetTopology->getPointDataArray().setParent(&fixedCloudDOF->x);

  return fixedCloudDOF;
}

// ---------------------------------------------------------------------
MechanicalObject<Vec3Types>::SPtr loadWarpedPointCloud(
  Node * parentNode,
  itk::PointSet<float,3> *fixedPointCloud,
  itk::PointSet<float,3> *movingPointCloud,
  unsigned int numPoints = 10,
  bool Verbose = false
  )
{
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create warped point mechanical object..." << std::endl;
    }
    typedef itk::PointSet<float,3> PointCloudType;
    typedef PointCloudType::PointIdentifier PointIdentifier;
    typedef PointCloudType::PointsContainer PointsContainer;
    typedef PointCloudType::PointType PointType;
    
    itk::PointsLocator<PointCloudType::PointsContainer>::Pointer pointsLocator = 
        itk::PointsLocator<PointCloudType::PointsContainer>::New();
    pointsLocator->SetPoints(movingPointCloud->GetPoints());
    pointsLocator->Initialize();
    
    PointCloudType::PointIdentifier size = fixedPointCloud->GetNumberOfPoints();

    PointsContainer::Pointer outputPoints = PointsContainer::New();
    outputPoints->resize(size);
    for(PointCloudType::PointIdentifier i = 0; i < size; ++i)
    {
        std::vector<PointCloudType::PointIdentifier> points;
        PointType dx, fixedPoint, movingPoint; 
        
        fixedPoint = fixedPointCloud->GetPoint(i);
        pointsLocator->FindClosestNPoints(fixedPoint,numPoints,points);
        dx.Fill(0);
        for(size_t k = 0; k < points.size(); ++k)
        {
            PointType p = movingPointCloud->GetPoint(points[k]);
            dx += (p-fixedPoint );
        }
        float scale = 1.0f/points.size();
        dx[0]*=scale;
        dx[1]*=scale;
        dx[2]*=scale;
        movingPoint[0] = fixedPoint[0]+dx[0];
        movingPoint[1] = fixedPoint[1]+dx[1];
        movingPoint[2] = fixedPoint[2]+dx[2];
        outputPoints->SetElement(i,movingPoint);
    }    
    
    MechanicalObject<Vec3Types>::SPtr movingCloudDOF =
    addNew<MechanicalObject<Vec3Types> >(parentNode, "movingCloudDOF");

    copyVertices(outputPoints,movingCloudDOF.get());
    
    // Create the Point Set Topology
    PointSetTopologyContainer::SPtr pointSetTopology =
    addNew<PointSetTopologyContainer>(parentNode, "Topology");
    pointSetTopology->getPointDataArray().setParent(&movingCloudDOF->x);
    
    return movingCloudDOF;
}

// ---------------------------------------------------------------------
MechanicalObject<Vec3Types>::SPtr loadWarpedPointCloud(
  Node * parentNode,
  itk::TranslationTransform< double,3>* transform,
  itk::PointSet<float,3> *fixedPointCloud,
  itk::PointSet<float,3> *movingPointCloud,
  bool Verbose
  )
{
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create warped point mechanical object..." << std::endl;
    }
  typedef itk::Image<unsigned char,3> BinaryImageType;
  typedef itk::PointSet<float,3> PointSetType;
  typedef PointSetType::PointsContainer PointsContainer;
  
  MechanicalObject<Vec3Types>::SPtr warpedCloudDOF =
    addNew<MechanicalObject<Vec3Types> >(parentNode, "warpedCloudDOF");

  // Get positions
  size_t numberOfPoints = fixedPointCloud->GetNumberOfPoints();
  if (Verbose)
    {
    std::cout << "Number of points: " << numberOfPoints << std::endl;
    }

  // Traverse fixedPointCloud nodes
  PointsContainer *points =
    fixedPointCloud->GetPoints();
  itk::TranslationTransform< double,3>::InputPointType  fixedPoint;
  itk::TranslationTransform< double,3>::OutputPointType movingPoint;
  itk::TranslationTransform< double,3>::InverseTransformBasePointer inverseTransform = transform->GetInverseTransform();

  warpedCloudDOF->resize(numberOfPoints);
  Data<MechanicalObject<Vec3Types>::VecCoord> *x =
    warpedCloudDOF->write(sofa::core::VecCoordId::position());

  BinaryImageType::IndexType index;
  MechanicalObject<Vec3Types>::VecCoord &vertices = *x->beginEdit();
  for(size_t i = 0; i < numberOfPoints; ++i)
    {
    fixedPoint  = points->GetElement(i);
    movingPoint = inverseTransform->TransformPoint(fixedPoint);
    vertices[i][0] = movingPoint[0];
    vertices[i][1] = movingPoint[1];
    vertices[i][2] = movingPoint[2];
    }
  x->endEdit();

  // Create the Point Set Topology
  PointSetTopologyContainer::SPtr pointSetTopology =
  addNew<PointSetTopologyContainer>(parentNode, "Topology");
  pointSetTopology->getPointDataArray().setParent(&warpedCloudDOF->x);

  return warpedCloudDOF;
}


/// Create a FEM in parentNode.  A MeshTopology should be defined in
/// parentNode prior to calling this function.
// ---------------------------------------------------------------------
void createFiniteElementModel(Node* parentNode, bool linearFEM,
                              const Vec3Types::Real &youngModulus,
                              const Vec3Types::Real &poissonRatio,
                              bool Verbose = false
                              )
{
    
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create finite element model..." << std::endl;
    }
  if (linearFEM)
    {
    if (Verbose)
      {
      std::cout << "Create linear FEM..." << std::endl;
      }
    TetrahedronFEMForceField< Vec3Types >::SPtr femSolver =
      addNew<TetrahedronFEMForceField< Vec3Types > >(parentNode,"femSolver");
    femSolver->setComputeGlobalMatrix(false);
    femSolver->setMethod("large");
    femSolver->setPoissonRatio(poissonRatio);
    femSolver->setYoungModulus(youngModulus);
    femSolver->_computeVonMisesStress.setValue(2);
    return;
    }
  if (Verbose)
    {
    std::cout << "Create non-linear FEM..." << std::endl;
    }
  Node::SPtr                   behavior = parentNode->createChild("behavior");
  MechanicalObject<Vec3Types> *tetMesh  =
    dynamic_cast<MechanicalObject<Vec3Types>*>( parentNode->getMechanicalState());

  // Define control nodes
  BarycentricShapeFunction<ShapeFunction3>::SPtr shapeFunction =
    addNew<BarycentricShapeFunction<ShapeFunction3> >(parentNode,
      "shapeFunction");

  // Sample mesh where the deformation gradients re going to be defined
  TopologyGaussPointSampler::SPtr sampler = addNew<TopologyGaussPointSampler>(
    behavior,"sampler");
  sampler->f_inPosition.setParent(&tetMesh->x);

  MechanicalObject<F331Types>::SPtr F = addNew<MechanicalObject<F331Types> >(
    behavior,"F");

  // Map mesh to sampled nodes
  LinearMapping<Vec3Types,F331Types>::SPtr linearMapping =
    addNew<LinearMapping<Vec3Types,F331Types> >(behavior,"linearMapping");
  linearMapping->setModels(tetMesh,F.get());

  // Create strain measurements
  Node::SPtr strainNode = behavior->createChild("strain");
    {
    MechanicalObject<U331Types>::SPtr U =
      addNew<MechanicalObject<U331Types> >(strainNode,"U");
    PrincipalStretchesMapping<F331Types,U331Types>::SPtr principalMapping =
      addNew<PrincipalStretchesMapping<F331Types,U331Types> >(strainNode,
        "principalMapping");
    principalMapping->threshold.setValue(0.6);
    principalMapping->asStrain.setValue(false);
    principalMapping->setModels(F.get(),U.get());

    StabilizedNeoHookeanForceField<U331Types>::SPtr forceField =
      addNew<StabilizedNeoHookeanForceField<U331Types> >(strainNode,
        "Force Field");

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
MechanicalObject<Vec3Types>::SPtr loadMesh(Node* parentNode,
                                           itk::Mesh<float,3> *polyMesh,
                                           bool Verbose
                                           )
{
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create mechanical object with tetrahedral mesh..." << std::endl;
    }
  typedef itk::Mesh<float,3>::CellType CellType;
  typedef itk::Mesh<float,3>::CellsContainer::ConstIterator CellIterator;
  typedef itk::TetrahedronCell< CellType >  TetrahedronType;

  // load mesh
  itk::Mesh<float,3>::PointsContainer *points = polyMesh->GetPoints();
  itk::Mesh<float,3>::CellsContainer * tetras = polyMesh->GetCells();

  std::stringstream meshName;
  meshName << "Mesh";

  // Create mechanical object (dof) for the mesh and extract material parameters
  MechanicalObject<Vec3Types>::SPtr mechanicalMesh =
    addNew<MechanicalObject<Vec3Types> >(parentNode,meshName.str());

  copyVertices(points,mechanicalMesh.get());

  // Create the MeshTopology
  MeshTopology::SPtr meshTopology =
    addNew<MeshTopology>(parentNode, "Topology");
  meshTopology->seqPoints.setParent(&mechanicalMesh->x);

  // Copy tetrahedra array from vtk cell array
  MeshTopology::SeqTetrahedra& tetrahedra =
    *meshTopology->seqTetrahedra.beginEdit();

  tetrahedra.reserve(polyMesh->GetNumberOfCells());

  if (Verbose)
    {
    std::cout << "Total # of tetrahedra: " << polyMesh->GetNumberOfCells() << std::endl;
    }
  CellIterator cellIterator = tetras->Begin();
  CellIterator cellEnd      = tetras->End();
  while(cellIterator != cellEnd)
    {
    CellType *                 cell      = cellIterator.Value();
    TetrahedronType *          tet       = static_cast<TetrahedronType*>(cell);
    const itk::IdentifierType *vertexIds = tet->GetPointIds();
    tetrahedra.push_back(MeshTopology::Tetra(vertexIds[0], vertexIds[1],
        vertexIds[2], vertexIds[3]));
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
void computeVonMisesStress(TetrahedronFEMForceField<Vec3Types>* FemSolver,
    sofa::helper::vector<Vec3Types::Real> &vonMisesStress,
    sofa::helper::vector<Vec<6,Vec3Types::Real> > &strain,
    sofa::helper::vector<VecNoInit<6,Vec3Types::Real> > &stress,
    VecCoord3 &displacements, bool Verbose
    
)
{
    if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Computing stresses..." << std::endl;
    }
    MechanicalObject3* mechanicalObject;
    FemSolver->getContext()->get(mechanicalObject);
    
    BaseMeshTopology* mesh;
    FemSolver->getContext()->get(mesh);
    
    const VecCoord3& X = *mechanicalObject->getX();
    ReadAccessor<Data<VecCoord3> > X0 =  FemSolver->_initialPoints;

    for (size_t i = 0; i < X0.size(); i++)
        displacements[i] = (X[i] - X0[i]);

//     std::cout << "Displ = " << displacements << std::endl;
    
    BaseMeshTopology::SeqTetrahedra::const_iterator it;
    size_t el;

    for(it = mesh->getTetrahedra().begin(), el = 0 ; it != mesh->getTetrahedra().end() ; ++it, ++el)
    {
        Vec<6,Vec3Types::Real> vStrain;
        Mat<3,3,Vec3Types::Real> gradU;
        Mat<4,4,Vec3Types::Real>& shf = FemSolver->elemShapeFun[el];

        /// compute gradU
        for (size_t k = 0; k < 3; ++k) {
            for (size_t l = 0; l < 3; ++l)  {
                gradU[k][l] = 0.0;
                for (size_t m = 0; m < 4; ++m)
                    gradU[k][l] += shf[l+1][m] * displacements[(*it)[m]][k];
            }
        }
//         std::cout << "gradU = " << gradU<< std::endl;

        Mat<3,3,Vec3Types::Real> strainMatrix = ((Vec3Types::Real)0.5)*(gradU + gradU.transposed() + gradU.transposed()*gradU);

        // Voigt notation for strain tensor
        for (size_t i = 0; i < 3; ++i)
            vStrain[i] = strainMatrix[i][i];
        
        vStrain[3] = strainMatrix[1][2];
        vStrain[4] = strainMatrix[0][2];
        vStrain[5] = strainMatrix[0][1];

        Vec3Types::Real lambda = FemSolver->elemLambda[el];
        Vec3Types::Real mu = FemSolver->elemMu[el];

        /// stress
        VecNoInit<6,Vec3Types::Real> s;

        Vec3Types::Real traceStrain = 0.0;
        for (size_t k = 0; k < 3; ++k) {
            traceStrain += vStrain[k];
            s[k] = vStrain[k]*2*mu;
        }

        for (size_t k = 3; k < 6; ++k)
            s[k] = vStrain[k]*2*mu;

        for (size_t k = 0; k < 3; ++k)
            s[k] += lambda*traceStrain;

        Vec3Types::Real vM;
        vM = sofa::helper::rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] - s[0]*s[1] - s[1]*s[2] - s[2]*s[0] + 3*s[3]*s[3] + 3*s[4]*s[4] + 3*s[5]*s[5]);
        if (vM < 1e-10)
            vM = 0.0;
        
        vonMisesStress[el]=(vM);
        stress[el]=(s);
        strain[el]=(vStrain);

//         std::cout << "VMStress: " << vM << std::endl;
    }
}
    
//------------------------------------------------------------------------------
void initMesh(vtkPolyData* outputPolyData, Node::SPtr skullMeshDOF, bool Verbose)
{
  MeshTopology *     topology = skullMeshDOF->getNodeObject<MeshTopology>();
  MechanicalObject3 *dof      =
    skullMeshDOF->getNodeObject<MechanicalObject3>();

  vtkNew<vtkPoints> points;
  const vtkIdType   numberOfPoints = topology->getNbPoints();
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
    vtkIdType    cell[4];
    cell[0] = t[0];
    cell[1] = t[1];
    cell[2] = t[2];
    cell[3] = t[3];
    cells->InsertNextCell(4, cell);
    }
  outputPolyData->SetPolys(cells.GetPointer());

  const VecDeriv3 &forces = *dof->getF();

  vtkNew<vtkFloatArray> forceVectors;
  forceVectors->SetNumberOfComponents(3);
  forceVectors->SetName("Forces");
  forceVectors->SetNumberOfTuples(forces.size());
  
  vtkNew<vtkFloatArray> displacements;
  displacements->SetName("Displacements");
  displacements->SetNumberOfComponents(1);
  displacements->SetNumberOfValues(forces.size());

  vtkNew<vtkFloatArray> vonMisesStress;
  vonMisesStress->SetNumberOfComponents(1);
  vonMisesStress->SetName("Von Mises Stress");
  vonMisesStress->SetNumberOfValues(topology->getNbTetras());
  
  vtkNew<vtkFloatArray> stressEigenvalues;
  stressEigenvalues->SetNumberOfComponents(3);
  stressEigenvalues->SetName("Stress Eigenvalues");
  stressEigenvalues->SetNumberOfValues(topology->getNbTetras());
  
  vtkNew<vtkFloatArray> strainEigenvalues;
  strainEigenvalues->SetNumberOfComponents(3);
  strainEigenvalues->SetName("Strain Eigenvalues");
  strainEigenvalues->SetNumberOfValues(topology->getNbTetras());

  TetrahedronFEMForceField<Vec3Types>* FEM = dynamic_cast<TetrahedronFEMForceField<Vec3Types>*>(skullMeshDOF->getObject("SkullMesh_femSolver"));
    
  sofa::helper::vector<Vec3Types::Real> vmStress;
  vmStress.resize(topology->getNbTetras());
  sofa::helper::vector<Vec<6,Vec3Types::Real> > strain;
  strain.resize(topology->getNbTetras());
  sofa::helper::vector<VecNoInit<6,Vec3Types::Real> > stress;
  stress.resize(topology->getNbTetras());
  sofa::helper::vector<Vec<3,Vec3Types::Real> > dx;
  dx.resize(forces.size());

  computeVonMisesStress(FEM,vmStress,strain,stress,dx,Verbose);
  
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Populating VTK point data..." << std::endl;
    }
  for(size_t i = 0, end = forces.size(); i< end; ++i)
    {
    float f[3] = {0};
    f[0] = forces[i][0];
    f[1] = forces[i][1];
    f[2] = forces[i][2];
  
    forceVectors->SetTupleValue(i,f);
    displacements->SetValue(i,dx[i].norm());
  
    }
  
  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Populating VTK cell data..." << std::endl;
    std::cout << "Num tetras = " << topology->getNbTetras() << std::endl;
    }
    
  Eigen::EigenSolver<Eigen::Matrix3d> eigenSolver;
  for(int i = 0, end = topology->getNbTetras(); i < end; ++i)
  {
    vonMisesStress->SetValue(i,vmStress[i]);
        
    // Voigt tensor notation to matrix
    Eigen::Matrix3d stressMatrix, strainMatrix;
    stressMatrix << stress[i][0] , stress[i][5] , stress[i][4],
                    stress[i][5] , stress[i][1] , stress[i][3],
                    stress[i][4] , stress[i][3] , stress[i][2];
                    
    strainMatrix << strain[i][0] , strain[i][5] , strain[i][4],
                    strain[i][5] , strain[i][1] , strain[i][3],
                    strain[i][4] , strain[i][3] , strain[i][2];              
                    
//     std::cout << stressMatrix << std::endl;
//     std::cout << strainMatrix << std::endl;
    eigenSolver.compute(stressMatrix,false);
    
    const Eigen::Vector3cd &eigenvalues = eigenSolver.eigenvalues();
//     std::cout << eigenvalues << std::endl;
    
    stressEigenvalues->SetTuple3(i,eigenvalues[0].real(),eigenvalues[1].real(),eigenvalues[2].real());
//     
//     eigenSolver.compute(strainMatrix,false);
//     
//     eigenvalues = eigenSolver.eigenvalues();
//     
//     e[0] = eigenvalues[0].real();
//     e[1] = eigenvalues[1].real();
//     e[2] = eigenvalues[2].real();
//         
//     strainEigenvalues->SetTupleValue(i,e);
    
  }

    if (Verbose)
    {
    std::cout << "************************************************************"
                << std::endl;
    std::cout << "Adding Point Data Arrays..." << std::endl;
    }
  outputPolyData->GetPointData()->AddArray(forceVectors.GetPointer());
  outputPolyData->GetPointData()->AddArray(displacements.GetPointer());


    if (Verbose)
    {
    std::cout << "************************************************************"
                << std::endl;
    std::cout << "Adding cell Data Arrays..." << std::endl;
    }
  outputPolyData->GetCellData()->AddArray(vonMisesStress.GetPointer());
//   outputPolyData->GetCellData()->AddArray(stressEigenvalues.GetPointer());
//   outputPolyData->GetCellData()->AddArray(strainEigenvalues.GetPointer());
}

//------------------------------------------------------------------------------
double meanSquareError(MechanicalObject<Vec3Types>::SPtr mesh1,
                       MechanicalObject<Vec3Types>::SPtr mesh2,
                       const std::vector<itk::PointSet<float,3>::PointIdentifier> &indexMap
                      )
{
  const Data<MechanicalObject<Vec3Types>::VecCoord>* position1 =
    mesh1->read(VecCoordId::position());
  const Data<MechanicalObject<Vec3Types>::VecCoord>* position2 =
    mesh2->read(VecCoordId::position());

  if (!position1 || !position2)
    {
    std::cerr << "No positions: " << position1 << ", " << position2 <<
      std::endl;
    return -1.;
    }
  // Copy vertices from vtk mesh
  const MechanicalObject<Vec3Types>::VecCoord& vertices1 =
    position1->getValue();
  const MechanicalObject<Vec3Types>::VecCoord& vertices2 =
    position2->getValue();

  size_t numberOfPoints = indexMap.size();
  if (numberOfPoints != vertices2.size())
    {
    std::cerr << "Not the same number of vertices: "
              << vertices2.size() << " != " << indexMap.size() << std::endl;
    return -1.;
    }

  double                                                error = 0.;

  for(size_t i = 0, j = 0; j < numberOfPoints ; ++i, ++j)
    {
    Vector3 dx = vertices1[indexMap[j]] - vertices2[j];
    error += dx.norm2() / numberOfPoints;
    }
  return error;
}

itk::TranslationTransform< double,3>::Pointer
registerPointClouds(itk::PointSet<float,3> *fixedMesh,
                    itk::PointSet<float,3> *movingMesh,
                    const int &numberOfIterations,
                    const double &gradientTolerance,
                    const double &valueTolerance,
                    const double &epsilonFunction,
                    bool Verbose = false)
{
    if (Verbose)
    {
    std::cout << "************************************************************"
                << std::endl;
    std::cout << "Register meshes using ICP..." << std::endl;
    }
  typedef itk::PointSet<float,3> PointSetType;
  if (Verbose)
    {
    std::cout << "Number of moving Points = " <<
      movingMesh->GetNumberOfPoints( ) << std::endl;
    }
  if (Verbose)
    {
    std::cout << "Number of fixed Points = " <<
      fixedMesh->GetNumberOfPoints() << std::endl;
    }
  //-----------------------------------------------------------
  // Set up  the Metric
  //-----------------------------------------------------------
  typedef itk::EuclideanDistancePointMetric<PointSetType,
                                            PointSetType> MetricType;
  MetricType::Pointer metric = MetricType::New();
  if (Verbose)
    {
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
  typedef itk::PointSetToPointSetRegistrationMethod<PointSetType,
                                                    PointSetType>
    RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

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
  bool useDistanceMap = false;
  if(useDistanceMap)
  {
    typedef itk::Image< unsigned char,  3 >  BinaryImageType;
    typedef itk::PointSetToImageFilter<PointSetType,
                                        BinaryImageType> PointsToImageFilterType;
    PointsToImageFilterType::Pointer pointsToImageFilter =
        PointsToImageFilterType::New();
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
    typedef itk::DanielssonDistanceMapImageFilter<BinaryImageType,
                                                    DistanceImageType>
        DistanceFilterType;
    DistanceFilterType::Pointer distanceFilter = DistanceFilterType::New();
    distanceFilter->SetInput( binaryImage );
    distanceFilter->Update();
    metric->SetDistanceMap( distanceFilter->GetOutput() );
  }
  // Connect an observer
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );
  if (Verbose)
    {
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
    std::cout << "registerMeshes(): Finished registration..." << std::endl;
    }
  return transform;
}

void mapFixedPointCloudToTetMesh(itk::PointSet<float,3> *pointCloud, 
                                 itk::PointSet<float,3> *mesh, 
                                 std::vector<itk::PointSet<float,3>::PointIdentifier> &indexMap, bool Verbose)
{
    typedef itk::PointSet<float,3> PointCloudType;
    
    itk::PointsLocator<PointCloudType::PointsContainer>::Pointer pointsLocator = 
        itk::PointsLocator<PointCloudType::PointsContainer>::New();
    pointsLocator->SetPoints(mesh->GetPoints());
    pointsLocator->Initialize();
    
    PointCloudType::PointIdentifier size = pointCloud->GetNumberOfPoints();
    
    indexMap.reserve(size);
    for(PointCloudType::PointIdentifier i = 0; i < size; ++i)
    {
        PointCloudType::PointIdentifier index = pointsLocator->FindClosestPoint(pointCloud->GetPoint(i));
        indexMap.push_back(index);
    }

  if (Verbose)
    {
    std::cout << "Map between mesh and fixed point cloud created" << std::endl;
    std::cout << "Total points = " << indexMap.size() << std::endl;  
    }
    
}

itk::TranslationTransform<double,
                          3>::Pointer loadTransform( std::string filename )
{
  itk::TranslationTransform<double,3>::Pointer transform;
  typedef itk::TransformFileReader TransformReaderType;
  typedef TransformReaderType::TransformListType TransformListType;

  TransformReaderType::Pointer transformReader = TransformReaderType::New();
  transformReader->SetFileName( filename );

  transformReader->Update();

  TransformListType * transforms =
    transformReader->GetTransformList();
  TransformListType::const_iterator transformIt = transforms->begin();
  while( transformIt != transforms->end() )
    {
    if( !strcmp( (*transformIt)->GetNameOfClass(), "TranslationTransform") )
      {
      transform =
        static_cast<itk::TranslationTransform<double,
                                              3> *>( (*transformIt).GetPointer() );
      }

    ++transformIt;
    }
    return transform;
}

//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  PARSE_ARGS;

  typedef float PixelType;
  typedef itk::PointSet<PixelType,3> PointCloudType;
  typedef itk::Mesh<PixelType,3> MeshType;
  typedef itk::VTKPointCloudReader<PointCloudType> PointCloudReaderType;

  typedef itk::VTKTetrahedralMeshReader<MeshType> MeshReaderType;

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

  // Read point cloud data
  PointCloudReaderType::Pointer vtkFixedPointCloudReader =
    PointCloudReaderType::New();
  vtkFixedPointCloudReader->SetFileName(FixedPointCloud);
  PointCloudReaderType::Pointer vtkMovingPointCloudReader =
    PointCloudReaderType::New();
  vtkMovingPointCloudReader->SetFileName(MovingPointCloud);

  if (Verbose)
    {
    std::cout << "Read data..." << std::endl;
    }
  try
    {
    vtkFixedPointCloudReader->Update();
    vtkMovingPointCloudReader->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr <<
      "Error during vtkFixedMeshReader->Update() and vtkMovingMeshReader->Update() "
              << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  if (Verbose)
    {
    std::cout << "Read mesh..." << std::endl;
    }

  if (!IsMeshInRAS)
    {
    std::cout<<"Mesh x,y coordinates will be inverted" << std::endl;
    }

  // Read tetrahedral mesh
  MeshReaderType::Pointer vtkTetMeshReader = MeshReaderType::New();
  vtkTetMeshReader->SetFileName(FixedImageMesh);

  try
    {
    std::cerr << "Trying to read..." << std::endl;
    vtkTetMeshReader->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr <<
      "Error during vtkTetMeshReader->Update() and vtkMovingMeshReader->Update() "
              << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  MeshType::Pointer       tetMesh       = vtkTetMeshReader->GetOutput();
  PointCloudType::Pointer fixedPointCloud = vtkFixedPointCloudReader->GetOutput();
  PointCloudType::Pointer movingPointCloud = vtkMovingPointCloudReader->GetOutput();

  itk::TranslationTransform<double,3>::Pointer transform;
  if(InputTransform.size() > 0)
    {
    transform = loadTransform(InputTransform);
    }
  else
    {
    transform =
      registerPointClouds(
        fixedPointCloud.GetPointer(),movingPointCloud.GetPointer(),
        Iterations,GradientTolerance,ValueTolerance,
        epsilonFunction,Verbose);

    if(OutputTransform.size()>0)
      {
      itk::TransformFileWriter::Pointer writer =
        itk::TransformFileWriter::New();
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
      }
    }

    if(transform.IsNull())
    {
        std::cerr << "Null Transform!" << std::endl;
        return EXIT_FAILURE;
    }

  // Create a scene node
  Node::SPtr sceneNode = root->createChild("FEMSimulation");

  // Time stepper 
  createEulerSolverNode(sceneNode.get(),TimeIntegratorType);

  Node::SPtr warpedCloudDOFNode                    = sceneNode->createChild(
    "WarpedDOFCloud");
  MechanicalObject<Vec3Types>::SPtr warpedCloudDOF = loadWarpedPointCloud(
    warpedCloudDOFNode.get(), fixedPointCloud, movingPointCloud,10u,Verbose);

  // Fix the oositions of these points
  FixedConstraint3::SPtr fixWarpedDOF = addNew<FixedConstraint3>(
      warpedCloudDOFNode.get(),"fixedContraint");
  fixWarpedDOF->f_fixAll.setValue(true);

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create Skull mesh..." << std::endl;
    }

  // Node for the mesh
  Node::SPtr skullDOFNode = sceneNode->createChild("SkullMesh");

  // Create mesh dof
  MechanicalObject<Vec3Types>::SPtr skullDOF = loadMesh(
    skullDOFNode.get(), tetMesh,Verbose);
  UniformMass3::SPtr skullMass          = addNew<UniformMass3>(
    skullDOFNode.get(),"Mass");
  skullMass->setTotalMass(100);

  // Add the finite element method to the node
  createFiniteElementModel(
    skullDOFNode.get(), true, youngModulus, poissonRatio, Verbose);
  
  // ROI
  Vector6 box;
  box[0] = 0;
  box[1] = 0;
  box[2] = 0;
  box[3] = 160;
  box[4] = 160;
  box[5] = 10;

  // Crete a fix contraint to fix the base of the skull
  BoxROI<Vec3Types>::SPtr boxRoi       = addNew<BoxROI<Vec3Types> >(
    skullDOFNode.get(),"SkullBaseRoi");
  boxRoi->boxes.beginEdit()->push_back(box);
  boxRoi->boxes.endEdit();
  boxRoi->p_drawBoxes.setValue(true);

  FixedConstraint3::SPtr fixedConstraint = addNew<FixedConstraint3>(
      skullDOFNode.get(),"fixedContraint");
  fixedConstraint->f_indices.setParent(&boxRoi->f_indices);

  if (Verbose)
    {
    std::cout << "************************************************************"
              << std::endl;
    std::cout << "Create force loads..." << std::endl;
    }
  std::vector<PointCloudType::PointIdentifier> indexMap;
  mapFixedPointCloudToTetMesh(fixedPointCloud,tetMesh,indexMap,Verbose);
  StiffSpringForceField<Vec3Types>::SPtr loads =
  sofa::core::objectmodel::New<StiffSpringForceField<Vec3Types> >(skullDOF.get(),warpedCloudDOF.get());
  loads->setName("Force Loads");
  skullDOFNode->addObject(loads);
  const VecCoord3 &meshPositions = *skullDOF->getX();
  const VecCoord3 &warpedCloudPositions = *warpedCloudDOF->getX();
  double distance = 1.;
  const vtkIdType numberOfPoints = indexMap.size();
  size_t sample = 0;
  for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
    {
    if (!(sample++ % 1))
      {
      distance = initialDistanceFraction*(meshPositions[indexMap[pointId]]-warpedCloudPositions[pointId]).norm();
      loads->addSpring(indexMap[pointId], pointId, pressureForce, 0.0, distance);
      }
    }

  const sofa::core::objectmodel::TagSet &tags = skullDOF->getTags();
  for (sofa::core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
    loads->addTag(*it);

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
    sofa::simulation::getSimulation()->exportXML(
      root.get(), sceneFileName.c_str());
    }
  if (Verbose)
    {
    std::cout << "Init..." << std::endl;
    }
  sofa::simulation::getSimulation()->init(root.get());

  if (Verbose)
    {
    std::cout << "Done Init..." << std::endl;
    }
  int    gluArgc = 1;
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
//     const size_t minimumNumberOfSteps = 150;

    double lastError = 1.;
    double stdDeviation = 0.;

    // We can't use the distance error directly because the simulation might
    // oscillate.
    for (size_t step = 0;stdDeviation > MinimumStandardDeviation ||
         (step < static_cast<size_t>(MaximumNumberOfSimulationSteps)) ; ++step)
      {
      sofa::simulation::getSimulation()->animate(root.get(), dt);
      //sofa::simulation::getSimulation()->animate(root.get());

      const double error = meanSquareError(skullDOF, warpedCloudDOF, indexMap);
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
  initMesh(posedSurface.GetPointer(), skullDOFNode,Verbose);
  
  if (Verbose)
    {
    std::cout << "Saving Polydata..." << std::endl;
    }
  bender::IOUtils::WritePolyData(posedSurface.GetPointer(), OutputTetMesh);

  if (Verbose)
    {
    std::cout << "Unload..." << std::endl;
    }
  sofa::simulation::getSimulation()->unload(root);

  return EXIT_SUCCESS;
}

