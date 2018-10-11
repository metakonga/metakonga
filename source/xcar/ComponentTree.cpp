/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtWidgets>
#include <QDebug>
//#include "treeitem.h"
#include "ComponentTree.h"

//! [0]
ComponentTree::ComponentTree(const QStringList &headers, QWidget *parent)
	: QTreeWidget(parent)
{
	setHeaderLabels(headers);
	components[BODY_COMPONENT] = new QTreeWidgetItem(this);
	components[BODY_COMPONENT]->setText(0, "Body Component");
	components[HARDPOINT_COMPONENT] = new QTreeWidgetItem(this);
	components[HARDPOINT_COMPONENT]->setText(0, "Hardpoint Component");
	components[FORCE_COMPONENT] = new QTreeWidgetItem(this);
	components[FORCE_COMPONENT]->setText(0, "Force Component");
	components[TEST_BED_COMPONENT] = new QTreeWidgetItem(this);
	components[TEST_BED_COMPONENT]->setText(0, "Test Bed Component");
	
// 	rootItem = new TreeItem(rootData);
	QTreeWidgetItem* child = new QTreeWidgetItem;
	child->setText(0, "Car Body");
	components[BODY_COMPONENT]->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Front Left Suspension");
	components[BODY_COMPONENT]->addChild(child);
	BCChild(child);
	FLS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Front Right Suspension");
	components[BODY_COMPONENT]->addChild(child);
	BCChild(child);
	FRS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Left Suspension");
	components[BODY_COMPONENT]->addChild(child);
	BCChild(child);
	RLS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Right Suspension");
	components[BODY_COMPONENT]->addChild(child);
	BCChild(child);
	RRS << child;

	child = new QTreeWidgetItem;
	child->setText(0, "Front Left Suspension");
	components[HARDPOINT_COMPONENT]->addChild(child);
	HCChild(child);
	FLS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Front Right Suspension");
	components[HARDPOINT_COMPONENT]->addChild(child);
	HCChild(child);
	FRS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Left Suspension");
	components[HARDPOINT_COMPONENT]->addChild(child);
	HCChild(child);
	RLS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Right Suspension");
	components[HARDPOINT_COMPONENT]->addChild(child);
	HCChild(child);
	RRS << child;

	child = new QTreeWidgetItem;
	child->setText(0, "Spring Damper");
	components[FORCE_COMPONENT]->addChild(child);
	FCChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Wheel Torque");
	components[FORCE_COMPONENT]->addChild(child);
	FCChild(child);

	child = new QTreeWidgetItem;
	child->setText(0, "Wheel");
	components[TEST_BED_COMPONENT]->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Vertical Link");
	components[TEST_BED_COMPONENT]->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Horizontal Link");
	components[TEST_BED_COMPONENT]->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Translational Driving");
	components[TEST_BED_COMPONENT]->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Rotational Driving");
	components[TEST_BED_COMPONENT]->addChild(child);
	components[TEST_BED_COMPONENT]->setHidden(true);
	connect(this, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(itemClick(QTreeWidgetItem*, int)));
 //	setupTree();
}
//! [0]

//! [1]
ComponentTree::~ComponentTree()
{
	//delete rootItem;
}

void ComponentTree::setFullCarComponent()
{
	components[BODY_COMPONENT]->setHidden(false);
	components[HARDPOINT_COMPONENT]->setHidden(false);
	components[FORCE_COMPONENT]->setHidden(false);
	foreach(QTreeWidgetItem* item, FRS)
	{
		item->setHidden(false);
	}
	foreach(QTreeWidgetItem* item, RLS)
	{
		item->setHidden(false);
	}
	foreach(QTreeWidgetItem* item, RRS)
	{
		item->setHidden(false);
	}
	components[TEST_BED_COMPONENT]->setHidden(true);
}

void ComponentTree::setQuarterCarComponent()
{
	setFullCarComponent();
	foreach(QTreeWidgetItem* item, FLS)
	{
		item->setHidden(false);
	}
	foreach(QTreeWidgetItem* item, FRS)
	{
		item->setHidden(true);
	}
	foreach(QTreeWidgetItem* item, RLS)
	{
		item->setHidden(true);
	}
	foreach(QTreeWidgetItem* item, RRS)
	{
		item->setHidden(true);
	}
}

void ComponentTree::setTestBedComponent()
{
	components[BODY_COMPONENT]->setHidden(true);
	components[HARDPOINT_COMPONENT]->setHidden(true);
	components[FORCE_COMPONENT]->setHidden(true);
	components[TEST_BED_COMPONENT]->setHidden(false);
}

void ComponentTree::itemClick(QTreeWidgetItem* item, int col)
{
	if (item->childCount())
		return;
	int sq = 0;
	QTreeWidgetItem* parent = item->parent();
	if (parent->parent())
	{
		parent = parent->parent();
		sq = 1;
	}
		

	
	QString text = parent->text(0);
	QString name;
	if (text == "Body Component")
		name += "B";
	else if (text == "Hardpoint Component")
		name += "H";
	else if (text == "Force Component")
		name += "F";
	else if (text == "Test Bed Component")
		name += "T";

	//text = item->parent()->text(0);// parent->child(0)->text(0);
	if (name == "B")
	{
		text = item->parent()->text(0);
		if (sq == 0)
			name += "_CB";
		else if (text == "Front Left Suspension")
			name += "_FLS";
		else if (text == "Front Right Suspension")
			name += "_FRS";
		else if (text == "Rear Left Suspension")
			name += "_RLS";
		else if (text == "Rear Right Suspension")
			name += "_RRS";

		if (item)
		{
			text = item->text(0);// parent->child(0)->child(0)->text(0);
			if (text == "Wheel")
				name += "_W";
			else if (text == "Knuckle")
				name += "_K";
			else if (text == "Upper Arm")
				name += "_U";
			else if (text == "Lower Arm")
				name += "_L";
			else if (text == "Steer Link")
				name += "_S";
		}
	}
	else if (name == "H")
	{
		text = item->parent()->text(0);
		if (text == "Front Left Suspension")
			name += "_FLS";
		else if (text == "Front Right Suspension")
			name += "_FRS";
		else if (text == "Rear Left Suspension")
			name += "_RLS";
		else if (text == "Rear Right Suspension")
			name += "_RRS";

		if (item)
		{
			text = item->text(0);
			if (text == "Wheel-Knuckle")
				name += "_WK";
			else if (text == "Knuckle-Upper Arm")
				name += "_KU";
			else if (text == "Knuckle-Lower Arm")
				name += "_KL";
			else if (text == "Knuckle-Steer Link")
				name += "_KS";
			else if (text == "Upper Arm-Car Body")
				name += "_UC";
			else if (text == "Lower Arm-Car Body")
				name += "_LC";
		}
	}
	else if (name == "F")
	{
		text = item->parent()->text(0);
		if (text == "Spring Damper")
			name += "_SD";
		else if (text == "Wheel Torque")
			name += "_WT";

		if (item)
		{
			text = item->text(0);
			if (text == "Front Left")
				name += "_FL";
			else if (text == "Front Right")
				name += "_FR";
			else if (text == "Rear Left")
				name += "_RL";
			else if (text == "Rear Right")
				name += "_RR";
		}
	}
	else if (name == "T")
	{
		text = item->text(0);
		if (text == "Wheel")
			name += "_W";
		else if (text == "Vertical Link")
			name += "_VL";
		else if (text == "Horizontal Link")
			name += "_HL";
		else if (text == "Translational Driving")
			name += "_TD";
		else if (text == "Rotational Driving")
			name += "_RD";
	}
	qDebug() << name;
}

void ComponentTree::BCChild(QTreeWidgetItem* parent)
{
	QTreeWidgetItem* child = new QTreeWidgetItem;
	child->setText(0, "Wheel");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Knuckle");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Upper Arm");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Lower Arm");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Steer Link");
	parent->addChild(child);
}

void ComponentTree::HCChild(QTreeWidgetItem* parent)
{
	QTreeWidgetItem* child = new QTreeWidgetItem;
	child->setText(0, "Wheel-Knuckle");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Knuckle-Upper Arm");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Knuckle-Lower Arm");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Knuckle-Steer Link");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Upper Arm-Car Body");
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Lower Arm-Car Body");
	parent->addChild(child);
}

void ComponentTree::FCChild(QTreeWidgetItem* parent)
{
	QTreeWidgetItem* child = new QTreeWidgetItem;
	child->setText(0, "Front Left");
	parent->addChild(child);
	FLS << child;
	child = new QTreeWidgetItem;
	child->setText(0, "Front Right");
	FRS << child;
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Left");
	RLS << child;
	parent->addChild(child);
	child = new QTreeWidgetItem;
	child->setText(0, "Rear Right");
	RRS << child;
	parent->addChild(child);
}
