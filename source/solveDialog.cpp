#include "solveDialog.h"
#include <QtWidgets>

solveDialog::solveDialog()
	: isDialogOk(false)
	, isCpu(true)
{
}



solveDialog::~solveDialog()
{
}

bool solveDialog::callDialog()
{
	QLabel *LSimTime = new QLabel("Simulation time");
	QLabel *LTimeStep = new QLabel("Time step");
	QLabel *LSaveStep = new QLabel("Save step");
	RBCpu = new QRadioButton("CPU", this);
	RBGpu = new QRadioButton("GPU", this);
	LESimTime = new QLineEdit;
	LETimeStep = new QLineEdit;
	LESaveStep = new QLineEdit;
	solveLayout = new QGridLayout;
	PBSolve = new QPushButton("Solve");
	PBCancel = new QPushButton("Cancel");
	RBCpu->setChecked(true);
	RBGpu->setChecked(false);
	solveLayout->addWidget(RBCpu, 0, 0);
	solveLayout->addWidget(RBGpu, 0, 1);
	solveLayout->addWidget(LSimTime, 1, 0);
	solveLayout->addWidget(LESimTime, 1, 1, 1, 2);
	solveLayout->addWidget(LTimeStep, 2, 0);
	solveLayout->addWidget(LETimeStep, 2, 1, 1, 2);
	solveLayout->addWidget(LSaveStep, 3, 0);
	solveLayout->addWidget(LESaveStep, 3, 1, 1, 2);
	solveLayout->addWidget(PBSolve, 4, 0);
	solveLayout->addWidget(PBCancel, 4, 1);

	
	this->setLayout(solveLayout);

	connect(PBSolve, SIGNAL(clicked()), this, SLOT(Click_Solve()));
	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_Cancel()));

	LESimTime->setText("0.1");
	LETimeStep->setText("1e-5");
	LESaveStep->setText("100");
	
	this->exec();
	
	if (isDialogOk){
		simTime = LESimTime->text().toDouble();
		timeStep = LETimeStep->text().toDouble();
		saveStep = LESaveStep->text().toDouble();
		isCpu = RBCpu->isChecked();
	}

	return isDialogOk;
}

void solveDialog::Click_Solve()
{
// 
// 	QStringList ss;
// 	if (LEWorldOrigin->text().split(",").size() == 3)
// 		ss = LEWorldOrigin->text().split(",");
// 	else
// 		if (LEWorldOrigin->text().split(" ").size() == 3)
// 			ss = LEWorldOrigin->text().split(" ");
// 		else
// 			if (LEWorldOrigin->text().split(", ").size() == 3)
// 				ss = LEWorldOrigin->text().split(", ");
// 			else {
// 				Object::msgBox("World origin is wrong data.", QMessageBox::Critical);
// 				return;
// 			}
// 	worldOrigin.x = ss.at(0).toDouble(); worldOrigin.y = ss.at(1).toDouble(); worldOrigin.z = ss.at(2).toDouble();
// 
// 	if (LEGridSize->text().split(",").size() == 3)
// 		ss = LEGridSize->text().split(",");
// 	else
// 		if (LEGridSize->text().split(" ").size() == 3)
// 			ss = LEGridSize->text().split(" ");
// 		else
// 			if (LEGridSize->text().split(", ").size() == 3)
// 				ss = LEGridSize->text().split(", ");
// 			else {
// 				Object::msgBox("Grid size is wrong data.", QMessageBox::Critical);
// 				return;
// 			}
// 	gridSize.x = ss.at(0).toUInt(); gridSize.y = ss.at(1).toUInt(); gridSize.z = ss.at(2).toUInt();

	this->close();
	isDialogOk = true;
}

void solveDialog::Click_Cancel()
{
	this->close();
	isDialogOk = false;
}