#include "planeDialog.h"
#include "mphysics_types.h"
#include "msgBox.h"
#include "checkFunctions.h"
#include "modeler.h"
#include "plane.h"
#include <QtWidgets>

planeDialog::planeDialog()
	: isDialogOk(false)
{
	LMaterial = new QLabel("Material");
	CBMaterial = new QComboBox;
	CBMaterial->addItems(getMaterialList());
}

planeDialog::~planeDialog()
{

}

plane* planeDialog::callDialog(modeler* md)
{
	QString nm;
	QTextStream(&nm) << "plane" << md->numPlane();
	QLabel *LName = new QLabel("Name");
	QLabel *LPa = new QLabel("Point a");
	QLabel *LPb = new QLabel("Point b");
	QLabel *LPc = new QLabel("Point c");
	QLabel *LPd = new QLabel("Point d");
	LEName = new QLineEdit(nm);
	LEName->setText(nm);
	LEPa = new QLineEdit;
	LEPb = new QLineEdit;
	LEPc = new QLineEdit;
	LEPd = new QLineEdit;
	QGridLayout *rectLayout = new QGridLayout;
	QPushButton *PBOk = new QPushButton("OK");
	QPushButton *PBCancel = new QPushButton("Cancel");
	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	rectLayout->addWidget(LMaterial, 0, 0);
	rectLayout->addWidget(CBMaterial, 0, 1, 1, 2);
	rectLayout->addWidget(LName, 1, 0);		rectLayout->addWidget(LEName, 1, 1, 1, 2);
	rectLayout->addWidget(LPa, 2, 0);		rectLayout->addWidget(LEPa, 2, 1, 1, 2);
	rectLayout->addWidget(LPb, 3, 0);		rectLayout->addWidget(LEPb, 3, 1, 1, 2);
	rectLayout->addWidget(LPc, 4, 0);		rectLayout->addWidget(LEPc, 4, 1, 1, 2);
	rectLayout->addWidget(LPd, 5, 0);		rectLayout->addWidget(LEPd, 5, 1, 1, 2);
	rectLayout->addWidget(PBOk, 6, 0);
	rectLayout->addWidget(PBCancel, 6, 1);
	this->setLayout(rectLayout);

	this->exec();

	plane *p = NULL;
	if (isDialogOk)
	{
		p = md->makePlane(LEName->text(), tMaterial(CBMaterial->currentIndex()), ROLL_BOUNDARY);
		QStringList ss = LEPa->text().split(" ");
		VEC3D xw(ss.at(0).toDouble(), ss.at(1).toDouble(), ss.at(2).toDouble());
		ss = LEPb->text().split(" ");
		VEC3D pa(ss.at(0).toDouble(), ss.at(1).toDouble(), ss.at(2).toDouble());
		ss = LEPd->text().split(" ");
		VEC3D pb(ss.at(0).toDouble(), ss.at(1).toDouble(), ss.at(2).toDouble());
		ss = LEPc->text().split(" ");
		VEC3D pc(ss.at(0).toDouble(), ss.at(1).toDouble(), ss.at(2).toDouble());
		p->define(xw, pa, pc, pb);
	}
	return p;
}

void planeDialog::Click_ok()
{
	if (LEPa->text().isEmpty() || LEPb->text().isEmpty() || LEPc->text().isEmpty() || LEPd->text().isEmpty()){
		msgBox("There is empty data!!", QMessageBox::Critical);
		return;
	}

	QStringList ss;// = LEPa->text().split(",");
	if (LEPa->text().split(" ").size() != 3)
	{
		msgBox("Point a is wrong data.", QMessageBox::Critical);
		return;
	}
			//points[0].x = ss.at(0).toDouble(); points[0].y = ss.at(1).toDouble(); points[0].z = ss.at(2).toDouble();

	if (LEPb->text().split(" ").size() != 3)
	{
		msgBox("Point b is wrong data.", QMessageBox::Critical);
		return;
	}
					//points[1].x = ss.at(0).toDouble(); points[1].y = ss.at(1).toDouble(); points[1].z = ss.at(2).toDouble();

	if (LEPc->text().split(" ").size() != 3)
	{
		msgBox("Point c is wrong data.", QMessageBox::Critical);
		return;
	}
							//points[2].x = ss.at(0).toDouble(); points[2].y = ss.at(1).toDouble(); points[2].z = ss.at(2).toDouble();

	if (LEPd->text().split(" ").size() != 3)
	{
		msgBox("Point d is wrong data.", QMessageBox::Critical);
		return;
	}
									//points[3].x = ss.at(0).toDouble(); points[3].y = ss.at(1).toDouble(); points[3].z = ss.at(2).toDouble();
///
									//mtype = material_str2enum(CBMaterial->currentText().toStdString());
									//material = getMaterialConstant(mtype);

									//name = LEName->text();

									//nRect++;
	this->close();
	//delete rectDialog; rectDialog = NULL;
	isDialogOk = true;
}

void planeDialog::Click_cancel()
{
	this->close();
	isDialogOk = false;
}