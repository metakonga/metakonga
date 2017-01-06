#include "cubeDialog.h"
#include "mphysics_types.h"
#include "msgBox.h"
#include "checkFunctions.h"
#include "modeler.h"
#include "cube.h"
#include <QtWidgets>


cubeDialog::cubeDialog()
	: isDialogOk(false)
{
	LMaterial = new QLabel("Material");
	CBMaterial = new QComboBox;
	CBMaterial->addItems(getMaterialList());
}

cubeDialog::~cubeDialog()
{
	//this->
}

cube* cubeDialog::callDialog(modeler *md)
{
	
		//cubeDialog = new QDialog;
	QString nm;
	QTextStream(&nm) << "cube" << md->numCube();
	this->setObjectName("Cube Dialog");
	LName = new QLabel("Name");
	LStartPoint = new QLabel("Start point");
	LEndPoint = new QLabel("End point");
	
	LEName = new QLineEdit;
	LEName->setText(nm);
	LEStartPoint = new QLineEdit;
	LEEndPoint = new QLineEdit;
	
	cubeLayout = new QGridLayout;
	
	PBOk = new QPushButton("OK");
	PBCancel = new QPushButton("Cancel");
	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	cubeLayout->addWidget(LMaterial, 0, 0);
	cubeLayout->addWidget(CBMaterial, 0, 1, 1, 2);
	cubeLayout->addWidget(LName, 1, 0);
	cubeLayout->addWidget(LEName, 1, 1, 1, 2);
	cubeLayout->addWidget(LStartPoint, 2, 0);
	cubeLayout->addWidget(LEStartPoint, 2, 1, 1, 2);
	cubeLayout->addWidget(LEndPoint, 3, 0);
	cubeLayout->addWidget(LEEndPoint, 3, 1, 1, 2);
	cubeLayout->addWidget(PBOk, 4, 0);
	cubeLayout->addWidget(PBCancel, 4, 1);
	this->setLayout(cubeLayout);
	
	this->exec();
	cube *c = NULL;
	if (isDialogOk)
	{
		c = md->makeCube(this->objectName(), tMaterial(CBMaterial->currentIndex()), ROLL_BOUNDARY);
		QStringList chList = LEStartPoint->text().split(" ");
		float minPoint[3] = { chList.at(0).toFloat(), chList.at(1).toFloat(), chList.at(2).toFloat() };
		chList = LEEndPoint->text().split(" ");
		float maxPoint[3] = { chList.at(0).toFloat(), chList.at(1).toFloat(), chList.at(2).toFloat() };
		c->define(VEC3F(minPoint), VEC3F(maxPoint));
	}

	return c;
}

void cubeDialog::Click_ok()
{
	if (LEStartPoint->text().isEmpty()){
		msgBox("Value of start point is empty!!", QMessageBox::Critical);
	}
	else if (LEEndPoint->text().isEmpty()){
		msgBox("Value of end point is empty!!", QMessageBox::Critical);
	}

	if (!checkParameter3(LEStartPoint->text())){
		msgBox("Start point is wrong data.", QMessageBox::Critical);
		return;
	}
	else if (!checkParameter3(LEEndPoint->text())){
		msgBox("End point is wrong data.", QMessageBox::Critical);
		return;
	}

	this->setObjectName(LEName->text());// LEName->text();
// 	mtype = material_str2enum(CBMaterial->currentText().toStdString());
// 	material = getMaterialConstant(mtype);

// 	QStringList chList = LEStartPoint->text().split(" ");
// 	minPoint[0] = chList.at(0).toFloat();
// 	minPoint[1] = chList.at(1).toFloat();
// 	minPoint[2] = chList.at(2).toFloat();
// 
// 	chList = LEEndPoint->text().split(" ");
// 	maxPoint[0] = chList.at(0).toFloat();
// 	maxPoint[1] = chList.at(1).toFloat();
// 	maxPoint[2] = chList.at(2).toFloat();
// 
// 	width = maxPoint[0] - minPoint[0];
// 	height = maxPoint[1] - minPoint[1];
// 	depth = maxPoint[2] - minPoint[2];
// 
// 	vertice[0] = minPoint[0];		   vertice[1] = minPoint[1];		   vertice[2] = minPoint[2];
// 	vertice[3] = minPoint[0];		   vertice[4] = minPoint[1] + height; vertice[5] = minPoint[2];
// 	vertice[6] = minPoint[0];		   vertice[7] = minPoint[1];		   vertice[8] = minPoint[2] + depth;
// 	vertice[9] = minPoint[0];		   vertice[10] = minPoint[1] + height; vertice[11] = minPoint[2] + depth;
// 	vertice[12] = minPoint[0] + width; vertice[13] = minPoint[1];		   vertice[14] = minPoint[2] + depth;
// 	vertice[15] = minPoint[0] + width; vertice[16] = minPoint[1] + height; vertice[17] = minPoint[2] + depth;
// 	vertice[18] = minPoint[0] + width; vertice[19] = minPoint[1];		   vertice[20] = minPoint[2];
// 	vertice[21] = minPoint[0] + width; vertice[22] = minPoint[1] + height; vertice[23] = minPoint[2];
// 
// 	this->close();
// 
// 	//delete cubeDialog;
// 	//cubeDialog = NULL;
	this->close();
	isDialogOk = true;
}

void cubeDialog::Click_cancel()
{
	this->close();
	isDialogOk = false;
}