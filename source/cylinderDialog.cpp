#include "cylinderDialog.h"
#include "cylinder.h"
#include "mphysics_types.h"
#include "msgBox.h"
#include "checkFunctions.h"
#include "modeler.h"
#include <QtWidgets>


cylinderDialog::cylinderDialog()
	: isDialogOk(false)
{
	
}

cylinderDialog::~cylinderDialog()
{
	//this->
}

cylinder* cylinderDialog::callDialog(modeler *md)
{
	QLabel LMaterial("Material");
	QComboBox CBMaterial;
	CBMaterial.addItems(getMaterialList());
	QString nm;
	QTextStream(&nm) << "cylinder" << md->numCube();
	this->setObjectName("Cube Dialog");
	QLabel LName("Name");
	QLabel LBaseRadius("base radius");
	QLabel LTopRadius("top radius");
	//QLabel LLength("length");
	QLabel LTopPos("top position");
	QLabel LBasePos("base position");

	LEName = new QLineEdit;
	LEName->setText(nm);
	LEBaseRadius = new QLineEdit;
	LETopRadius = new QLineEdit;
	//LELength = new QLineEdit;
	LEBasePos = new QLineEdit;
	LETopPos = new QLineEdit;

	QGridLayout cylinderLayout;

	QPushButton PBOk("OK");
	QPushButton PBCancel("Cancel");
	connect(&PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(&PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	cylinderLayout.addWidget(&LMaterial, 0, 0);
	cylinderLayout.addWidget(&CBMaterial, 0, 1, 1, 2);
	cylinderLayout.addWidget(&LName, 1, 0);
	cylinderLayout.addWidget(LEName, 1, 1, 1, 2);
	cylinderLayout.addWidget(&LBaseRadius, 2, 0);
	cylinderLayout.addWidget(LEBaseRadius, 2, 1, 1, 2);
	cylinderLayout.addWidget(&LTopRadius, 3, 0);
	cylinderLayout.addWidget(LETopRadius, 3, 1, 1, 2);
// 	cylinderLayout.addWidget(&LLength, 4, 0);
// 	cylinderLayout.addWidget(LELength, 4, 1, 1, 2);
	cylinderLayout.addWidget(&LBasePos, 4, 0);
	cylinderLayout.addWidget(LEBasePos, 4, 1, 1, 2);
	cylinderLayout.addWidget(&LTopPos, 5, 0);
	cylinderLayout.addWidget(LETopPos, 5, 1, 1, 2);
	cylinderLayout.addWidget(&PBOk, 6, 0);
	cylinderLayout.addWidget(&PBCancel, 6, 1);
	this->setLayout(&cylinderLayout);

	this->exec();
	cylinder *cy = NULL;
	if (isDialogOk)
	{
		cy = md->makeCylinder(this->objectName(), tMaterial(CBMaterial.currentIndex()), ROLL_BOUNDARY);
		float br = LEBaseRadius->text().toDouble();
		float tr = LETopRadius->text().toDouble();
		//float len = LELength->text().toDouble();
		QStringList chList = LEBasePos->text().split(" ");
		VEC3D bpos = { chList.at(0).toDouble(), chList.at(1).toDouble(), chList.at(2).toDouble() };
		chList = LETopPos->text().split(" ");
		VEC3D tpos = { chList.at(0).toDouble(), chList.at(1).toDouble(), chList.at(2).toDouble() };
		cy->define(br, tr, bpos, tpos);
	}

	return cy;
}

void cylinderDialog::Click_ok()
{
// 	if (LEStartPoint->text().isEmpty()){
// 		msgBox("Value of start point is empty!!", QMessageBox::Critical);
// 	}
// 	else if (LEEndPoint->text().isEmpty()){
// 		msgBox("Value of end point is empty!!", QMessageBox::Critical);
// 	}
// 
// 	if (!checkParameter3(LEStartPoint->text())){
// 		msgBox("Start point is wrong data.", QMessageBox::Critical);
// 		return;
// 	}
// 	else if (!checkParameter3(LEEndPoint->text())){
// 		msgBox("End point is wrong data.", QMessageBox::Critical);
// 		return;
// 	}

	this->setObjectName(LEName->text());// LEName->text();

	this->close();
	isDialogOk = true;
}

void cylinderDialog::Click_cancel()
{
	this->close();
	isDialogOk = false;
}