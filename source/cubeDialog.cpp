#include "cubeDialog.h"
#include "msgBox.h"
#include "cube.h"
#include <QtWidgets>
#include "types.h"

cubeDialog::cubeDialog(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);
	CB_Type->addItems(getMaterialList());
	int mt = CB_Type->currentIndex();
	cmaterialType cmt = getMaterialConstant(mt);
	LE_Youngs->setText(QString("%1").arg(cmt.youngs));
	LE_PoissonRatio->setText(QString("%1").arg(cmt.poisson));
	LE_Density->setText(QString("%1").arg(cmt.density));
	LE_ShearModulus->setText(QString("%1").arg(cmt.shear));
	name = "Cube" + QString("%1").arg(cube::Number());
	LE_Name->setText(name);
	LE_StartPoint->setText("0.0 0.0 0.0");
	LE_EndPoint->setText("0.2 0.2 0.2");
	LE_Youngs->setReadOnly(true);
	LE_PoissonRatio->setReadOnly(true);
	LE_Density->setReadOnly(true);
	LE_ShearModulus->setReadOnly(true);
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	connect(CB_Type, SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBox(int)));
}

cubeDialog::~cubeDialog()
{
	//this->
}

void cubeDialog::changeComboBox(int idx)
{
	
	if (idx == USER_INPUT)
	{
		LE_Youngs->setReadOnly(false);
		LE_PoissonRatio->setReadOnly(false);
		LE_Density->setReadOnly(false);
		LE_ShearModulus->setReadOnly(false);
	}
	else
	{
		cmaterialType cmt;
		cmt = getMaterialConstant(idx);
		LE_Youngs->setText(QString("%1").arg(cmt.youngs));
		LE_PoissonRatio->setText(QString("%1").arg(cmt.poisson));
		LE_Density->setText(QString("%1").arg(cmt.density));
		LE_ShearModulus->setText(QString("%1").arg(cmt.shear));
		LE_Youngs->setReadOnly(true);
		LE_PoissonRatio->setReadOnly(true);
		LE_Density->setReadOnly(true);
		LE_ShearModulus->setReadOnly(true);
	}
}

void cubeDialog::Click_ok()
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

	name = LE_Name->text();
	type = CB_Type->currentIndex();
	youngs = LE_Youngs->text().toDouble();
	poisson = LE_PoissonRatio->text().toDouble();
	density = LE_Density->text().toDouble();
	shear = LE_ShearModulus->text().toDouble();
	/*c = md->makeCube(this->objectName(), tMaterial(CBMaterial->currentIndex()), ROLL_BOUNDARY);*/
	QStringList chList = LE_StartPoint->text().split(" ");
	start = VEC3D(chList.at(0).toDouble(), chList.at(1).toDouble(), chList.at(2).toDouble());
	chList = LE_EndPoint->text().split(" ");
	end = VEC3D(chList.at(0).toDouble(), chList.at(1).toDouble(), chList.at(2).toDouble());
	/*c->define(start);*/

	this->close();
	this->setResult(QDialog::Accepted);
}

void cubeDialog::Click_cancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}