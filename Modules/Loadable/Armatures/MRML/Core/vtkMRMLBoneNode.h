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

#ifndef __vtkMRMLBoneNode_h
#define __vtkMRMLBoneNode_h

// VTK includes
#include <vtkStdString.h>

// Slicer includes
#include <vtkMRMLAnnotationNode.h>

// Bender includes
#include <vtkBoneWidget.h>
#include <vtkQuaternion.h>

// Armatures includes
#include "vtkBenderArmaturesModuleMRMLCoreExport.h"
class vtkBoneRepresentation;
class vtkCallbackCommand;
class vtkObserverManager;
class vtkMRMLBoneDisplayNode;
class vtkMRMLDisplayNode;
class vtkMRMLAnnotationHierarchyNode;

/// \ingroup Bender_MRML
/// \brief Annotation to design and edit a bone.
///
/// In comparison with annotation nodes, vtkMRMLBoneNode only supports
/// vtkMRMLBoneDisplayNode for display nodes.
/// \sa vtkMRMLBoneDisplayNode, vtkMRMLArmatureNode
class VTK_BENDER_ARMATURES_MRML_CORE_EXPORT vtkMRMLBoneNode
  : public vtkMRMLAnnotationNode
{
public:
  //--------------------------------------------------------------------------
  // VTK methods
  //--------------------------------------------------------------------------

  static vtkMRMLBoneNode *New();
  vtkTypeMacro(vtkMRMLBoneNode,vtkMRMLAnnotationNode);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  /// Instantiate a bone node.
  virtual vtkMRMLNode* CreateNodeInstance();

  /// Get node XML tag name (like Volume, Model).
  virtual const char* GetNodeTagName() {return "Bone";};

  /// Read node attributes from XML file.
  virtual void ReadXMLAttributes( const char** atts);

  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  /// Copy the node's attributes to this object.
  virtual void Copy(vtkMRMLNode *node);

  /// Update references from scene.
  virtual void UpdateScene(vtkMRMLScene *scene);

  /// Alternative method to propagate events generated by observed nodes.
  virtual void ProcessMRMLEvents(vtkObject* caller,
                                 unsigned long event,
                                 void* callData);

  //--------------------------------------------------------------------------
  // Annotation methods
  //--------------------------------------------------------------------------
  virtual void Initialize(vtkMRMLScene* vtkNotUsed(mrmlScene))
    {
    vtkErrorMacro("Invalid method for vtkMRMLBoneNode."
      "Use Initialize(scene, parent) instead.");
    }

  void Initialize(
    vtkMRMLScene* mrmlScene, vtkMRMLAnnotationHierarchyNode* parent);

  /// Reimplement the vtkMRMLAnnotationNode method to create a modified event.
  //void SetVisible(int visible);

  //--------------------------------------------------------------------------
  // Bone methods
  //--------------------------------------------------------------------------

  /// Utility function that returns the associated display node as a bone
  /// display node.
  /// \sa GetDisplayNode()
  vtkMRMLBoneDisplayNode* GetBoneDisplayNode();
  /// Create a default display node if not already present.
  /// \sa CreateDefaultStorageNode()
  void CreateBoneDisplayNode();

  /// Convenience method to get the hierarchy node associated with this bone.
  vtkMRMLAnnotationHierarchyNode* GetHierarchyNode();

  /// Get the bone length.
  double GetLength();

  /// Set the bone name.
  virtual void SetName(const char* name);

  /// Set/Get the bone roll.
  void SetWidgetState(int state);
  int GetWidgetState();

  /// Mirroring the vtkBoneWidget widget states.
  //BTX
  enum WidgetStateType
    {
    PlaceHead = vtkBoneWidget::PlaceHead,
    PlaceTail = vtkBoneWidget::PlaceTail,
    Rest = vtkBoneWidget::Rest,
    Pose = vtkBoneWidget::Pose
    };
  //ETX

  /// Set/Get the bone representation.
  void SetBoneRepresentation(vtkBoneRepresentation* rep);
  vtkBoneRepresentation* GetBoneRepresentation();

  /// Helper function to set the representation.
  /// 1 for vtkCylinderBoneRepresentation, 2 for vtkDoubleConeBoneRepresentation,
  /// otherwise vtkBoneRepresentation.
  void SetBoneRepresentationType(int type); // \TO DO to logic
  vtkGetMacro(BoneRepresentationType, int);

  /// Set/Get the bone roll.
  void SetRoll(double roll);
  double GetRoll();

  /// Set/Get the head/tail position in the world coordinates.
  /// \sa GetWorldHeadRest(), GetWorldTailRest()
  /// \sa GetWorldHeadPose(), GetWorldTailPose()
  /// \sa SetWorldHeadRest(), SetWorldTailRest()
  void SetWorldHeadRest(const double* headPoint);
  void SetWorldTailRest(const double* tailPoint);
  double* GetWorldHeadRest();
  void GetWorldHeadRest(double head[3]);
  double* GetWorldHeadPose();
  void GetWorldHeadPose(double head[3]);
  double* GetWorldTailRest();
  void GetWorldTailRest(double tail[3]);
  double* GetWorldTailPose();
  void GetWorldTailPose(double tail[3]);

  /// Set/Get the local head/tail position in the parent coordinates.
  /// \sa GetLocalHeadRest(), GetLocalTailRest()
  /// \sa GetLocalHeadPose(), GetLocalTailPose()
  /// \sa SetLocalHeadRest(), SetLocalTailRest()
  void SetLocalHeadRest(const double* tailPoint);
  void SetLocalTailRest(const double* tailPoint);
  double* GetLocalHeadRest();
  void GetLocalHeadRest(double head[3]);
  double* GetLocalTailRest();
  void GetLocalHeadPose(double head[3]);
  double* GetLocalHeadPose();
  void GetLocalTailRest(double tail[3]);
  double* GetLocalTailPose();
  void GetLocalTailPose(double tail[3]);

  /// Set/Get the bone axes visibility.
  void SetShowAxes(int axesVisibility);
  int GetShowAxes();

  /// Set/Get the rest to pose rotation
  void SetRestToPoseRotation(double quad[4]);
  void GetRestToPoseRotation(double quad[4]);

  /// Set/Get the world to parent rotations.
  /// \sa GetWorldToParentRestRotation(), SetWorldToParentRestTranslation()
  /// \sa GetWorldToParentPoseRotation(), SetWorldToParentPoseTranslation()
  void SetWorldToParentRestRotation(const double* rotation);
  void SetWorldToParentPoseRotation(const double* rotation);
  vtkQuaterniond GetWorldToParentRestRotation();
  vtkQuaterniond GetWorldToParentPoseRotation();

  /// Set/Get the world to parent translations.
  /// \sa GetWorldToParentRestTranslation(), SetWorldToParentRestRotation()
  /// \sa GetWorldToParentPoseTranslation(), SetWorldToParentPoseRotation()
  void SetWorldToParentRestTranslation(const double* translation);
  void SetWorldToParentPoseTranslation(const double* translation);
  double* GetWorldToParentRestTranslation();
  double* GetWorldToParentPoseTranslation();
  void GetWorldToParentRestTranslation(double pos[3]);
  void GetWorldToParentPoseTranslation(double pos[3]);

  /// Get the parent to bone rotations.
  /// \sa GetParentToBoneRestRotation(), GetParentToBonePoseRotation()
  vtkQuaterniond GetParentToBoneRestRotation();
  vtkQuaterniond GetParentToBonePoseRotation();

  /// Get the parent to bone rotations.
  /// \sa GetParentToBoneRestTranslation(), GetParentToBonePoseTranslation()
  double* GetParentToBoneRestTranslation();
  double* GetParentToBonePoseTranslation();

  /// Get the world to bone rotations.
  /// \sa GetWorldToBoneHeadRestTranslation()
  /// \sa GetWorldToBoneTailRestTranslation()
  /// \sa GetWorldToBoneHeadPoseTranslation()
  /// \sa GetWorldToBoneTailPoseTranslation()
  /// \sa GetWorldToBoneRestRotation()
  /// \sa GetWorldToBonePoseRotation()
  vtkQuaterniond GetWorldToBoneRestRotation();
  vtkQuaterniond GetWorldToBonePoseRotation();

  /// Get the world to bone rotations.
  /// \sa GetWorldToBoneHeadRestTranslation()
  /// \sa GetWorldToBoneTailRestTranslation()
  /// \sa GetWorldToBoneHeadPoseTranslation()
  /// \sa GetWorldToBoneTailPoseTranslation()
  /// \sa GetWorldToBoneRestRotation()
  /// \sa GetWorldToBonePoseRotation()
  double* GetWorldToBoneHeadRestTranslation();
  double* GetWorldToBoneTailRestTranslation();
  double* GetWorldToBoneHeadPoseTranslation();
  double* GetWorldToBoneTailPoseTranslation();

  /// Set/Get the bone parenthood.
  void SetShowParenthood(int parenthood);
  int GetShowParenthood();

  /// Set/Get if the bone is linked with its parent.
  void SetBoneLinkedWithParent(bool parenthood);
  bool GetBoneLinkedWithParent();

  /// Set/Get if the bone has parent or not.
  vtkSetMacro(HasParent, bool);
  vtkGetMacro(HasParent, bool);

  // Rotate the tail in the parent coordinates system
  void RotateTailWithParentX(double angle);
  void RotateTailWithParentY(double angle);
  void RotateTailWithParentZ(double angle);
  void RotateTailWithParentWXYZ(double angle, double x, double y, double z);
  void RotateTailWithParentWXYZ(double angle, double axis[3]);

  // Rotate the tail in the world coordinates system
  void RotateTailWithWorldX(double angle);
  void RotateTailWithWorldY(double angle);
  void RotateTailWithWorldZ(double angle);
  void RotateTailWithWorldWXYZ(double angle, double x, double y, double z);
  void RotateTailWithWorldWXYZ(double angle, double axis[3]);

  // Scale the bone in Rest mode.
  void Scale(double factor);
  void Scale(double factorX, double factorY, double factorZ);
  void Scale(double factors[3]);

  // Translate the bone in Rest mode.
  void Translate(double x, double y, double z);
  void Translate(double rootHead[3]);

  // Rotate the bone in Rest mode.
  void RotateX(double angle);
  void RotateY(double angle);
  void RotateZ(double angle);
  void RotateWXYZ(double angle, double x, double y, double z);
  void RotateWXYZ(double angle, double axis[3]);

  // Transform the bone in Rest mode.
  void Transform(vtkTransform* t);

  // Set the bone's length in Rest mode.
  void SetLength(double size);

  //--------------------------------------------------------------------------
  // Helper methods
  //--------------------------------------------------------------------------

  /// Copy the properties of the widget into the node
  /// \sa PasteBoneNodeProperties()
  void CopyBoneWidgetProperties(vtkBoneWidget* boneWidget);

  /// Paste the properties of the node into the widget
  /// \sa CopyBoneWidgetProperties()
  void PasteBoneNodeProperties(vtkBoneWidget* boneWidget);

protected:
  vtkMRMLBoneNode();
  ~vtkMRMLBoneNode();

  vtkMRMLBoneNode(const vtkMRMLBoneNode&); /// not implemented
  void operator=(const vtkMRMLBoneNode&); /// not implemented

  /// MRML scene callback
  static void MRMLSceneCallback(
    vtkObject *caller, unsigned long eid,
    void *clientData, void *callData);

  virtual void ProcessMRMLSceneEvents(
    vtkObject *caller, unsigned long eid, void *callData);

  void AddBoneHierarchyNode();

  vtkMRMLAnnotationHierarchyNode* CurrentHierarchyNode;
  vtkObserverManager* SceneObserverManager;

  vtkCallbackCommand* Callback;

  vtkBoneWidget* BoneProperties;
  int BoneRepresentationType;
  bool LinkedWithParent;
  bool HasParent;
};

#endif
