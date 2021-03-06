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

#ifndef __qMRMLArmaturesAnimationWidget_h
#define __qMRMLArmaturesAnimationWidget_h

// Armatures includes
#include "qSlicerArmaturesModuleWidgetsExport.h"

class qMRMLArmaturesAnimationWidgetPrivate;
class vtkMRMLArmatureNode;
class vtkMRMLNode;

// SlicerQt includes
#include "qMRMLWidget.h"

// CTK includes
#include <ctkVTKObject.h>

/// \ingroup Slicer_QtModules_Armatures
class Q_SLICER_MODULE_ARMATURES_WIDGETS_EXPORT qMRMLArmaturesAnimationWidget
  : public qMRMLWidget
{
  Q_OBJECT
  QVTK_OBJECT
  /// This property controls whether the "select a frame" checkbox is checked
  /// by default when opening the file dialog. False by default.
  /// \sa selectFrame(), setSelectFrame()
  Q_PROPERTY(bool selectFrame READ selectFrame WRITE setSelectFrame)
public:

  typedef qMRMLWidget Superclass;
  qMRMLArmaturesAnimationWidget(QWidget *parent=0);
  virtual ~qMRMLArmaturesAnimationWidget();

  /// Return the selectFrame property value.
  /// \sa selectFrame
  bool selectFrame()const;

public slots:
  /// Set \a armatureNode as current.
  /// \sa setMRMLArmatureNode(vtkMRMLNode*) setMRMLNode(vtkMRMLNode* node)
  void setMRMLArmatureNode(vtkMRMLArmatureNode* armatureNode);
  /// Utility function to conveniently connect signals/slots.
  /// \sa setMRMLArmatureNode(vtkMRMLArmatureNode*)
  /// \sa setMRMLNode(vtkMRMLNode* node)
  void setMRMLArmatureNode(vtkMRMLNode* armatureNode);

  /// Open a dialog to select a BVH and potentially select a pose.
  void importAnimation();

  /// Set the selectFrame property value.
  /// \sa selectFrame
  void setSelectFrame(bool);
protected slots:
  void onFrameChanged(double frame);

  void updateWidgetFromArmatureNode();

protected:
  QScopedPointer<qMRMLArmaturesAnimationWidgetPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qMRMLArmaturesAnimationWidget);
  Q_DISABLE_COPY(qMRMLArmaturesAnimationWidget);
};

#endif
