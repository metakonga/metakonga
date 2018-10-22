#include "bodyInfoDialog.h"
#include "msgBox.h"
#include <QtWidgets>
#include "pointMass.h"

bodyInfoDialog::bodyInfoDialog(QWidget* parent)
	: QDialog(parent)
	, mt(0)
	, density(0)
	, youngs(0)
	, poisson(0)
	, shear(0)
{
	setupUi(this);
	CB_Material_Type->addItems(getMaterialList());
	LE_Position->setText("0.0, 0.0, 0.0");
	LE_Mass->setText("0.0");
	LE_Ixx->setText("0.0");
	LE_Iyy->setText("0.0");
	LE_Izz->setText("0.0");
	LE_Ixy->setText("0.0");
	LE_Izx->setText("0.0");
	LE_Iyz->setText("0.0");
	LE_Volume->setText("0.0");
	LE_Volume->setReadOnly(true);
	changeMaterialInputType(0);
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(Click_cancel()));
	connect(CB_Material_Input_Type, SIGNAL(currentIndexChanged(int)), this, SLOT(changeMaterialInputType(int)));
	connect(CB_Material_Type, SIGNAL(currentIndexChanged(int)), this, SLOT(changeMaterialType(int)));
}

bodyInfoDialog::~bodyInfoDialog()
{
	//this->
}

// pointMass* bodyInfoDialog::setBodyInfomation(object* o)
// {
// 	pointMass* pm = dynamic_cast<pointMass*>(o);
// 	mass = pm->Mass();
// 	volume = pm->Volume();
// 	material_type mt = pm->MaterialType();
// 	VEC3D loc = pm->Position();
// 	dx = pm->DiagonalInertia0().x;
// 	dy = pm->DiagonalInertia0().y;
// 	dz = pm->DiagonalInertia0().z;
// 	sx = pm->SymetricInertia0().x;
// 	sy = pm->SymetricInertia0().y;
// 	sz = pm->SymetricInertia0().z;
// 	x = loc.x; y = loc.y; z = loc.z;
// 	ixx = mass * dx; iyy = mass * dy; izz = mass * dz;
// 	ixy = mass * sx; iyz = mass * sy; izx = mass * sz;
// 	CB_Material_Type->setCurrentIndex(int(mt));
// 	LE_Position->setText(QString("%1, %2, %3").arg(loc.x).arg(loc.y).arg(loc.z));
// 	LE_Mass->setText(QString("%1").arg(mass));
// 	LE_Ixx->setText(QString("%1").arg(ixx));
// 	LE_Iyy->setText(QString("%1").arg(iyy));
// 	LE_Izz->setText(QString("%1").arg(izz));
// 	LE_Ixy->setText(QString("%1").arg(ixy));
// 	LE_Iyz->setText(QString("%1").arg(iyz));
// 	LE_Izx->setText(QString("%1").arg(izx));
// 	LE_Volume->setText(QString("%1").arg(volume));
// 	return pm;
// }

void bodyInfoDialog::setBodyInfomation(
	int mt, double x, double y, double z, 
	double _mass, double _vol, 
	double _ixx, double _iyy, double _izz, 
	double _ixy, double _iyz, double _izx)
{
	mass = _mass;
	volume = _vol;
	dx = ixx;
	dy = iyy;
	dz = izz;
	sx = ixy;
	sy = iyz;
	sz = izx;
	ixx = mass * dx; iyy = mass * dy; izz = mass * dz;
	ixy = mass * sx; iyz = mass * sy; izx = mass * sz;
	CB_Material_Type->setCurrentIndex(int(mt));
	LE_Position->setText(QString("%1 %2 %3").arg(x).arg(y).arg(z));
	LE_Mass->setText(QString("%1").arg(mass));
	LE_Ixx->setText(QString("%1").arg(ixx));
	LE_Iyy->setText(QString("%1").arg(iyy));
	LE_Izz->setText(QString("%1").arg(izz));
	LE_Ixy->setText(QString("%1").arg(ixy));
	LE_Iyz->setText(QString("%1").arg(iyz));
	LE_Izx->setText(QString("%1").arg(izx));
	LE_Volume->setText(QString("%1").arg(volume));
}

void bodyInfoDialog::changeMaterialInputType(int idx)
{
	if (idx == 0)
	{
		CB_Material_Type->setEnabled(true);
		CB_Material_Type->setCurrentIndex(CB_Material_Type->currentIndex());
		LE_Mass->setReadOnly(true);
		LE_Ixx->setReadOnly(true);
		LE_Iyy->setReadOnly(true);
		LE_Izz->setReadOnly(true);
		LE_Ixy->setReadOnly(true);
		LE_Iyz->setReadOnly(true);
		LE_Izx->setReadOnly(true);
	}
	else
	{
		CB_Material_Type->setEnabled(false);
		LE_Mass->setReadOnly(false);
		LE_Ixx->setReadOnly(false);
		LE_Iyy->setReadOnly(false);
		LE_Izz->setReadOnly(false);
		LE_Ixy->setReadOnly(false);
		LE_Iyz->setReadOnly(false);
		LE_Izx->setReadOnly(false);
	}
}

void bodyInfoDialog::changeMaterialType(int)
{
	cmaterialType cmt;
	cmt = getMaterialConstant(CB_Material_Type->currentIndex());
	mass = volume * cmt.density;
	LE_Mass->setText(QString("%1").arg(mass));
	LE_Ixx->setText(QString("%1").arg(mass * dx));
	LE_Iyy->setText(QString("%1").arg(mass * dy));
	LE_Izz->setText(QString("%1").arg(mass * dz));
	LE_Ixy->setText(QString("%1").arg(mass * sx));
	LE_Iyz->setText(QString("%1").arg(mass * sy));
	LE_Izx->setText(QString("%1").arg(mass * sz));
}

void bodyInfoDialog::Click_ok()
{
	cmaterialType cmt;
	mt = CB_Material_Type->currentIndex();
	cmt = getMaterialConstant(mt);
//	material_type mt = (material_type)mt;
	density = cmt.density;
	youngs = cmt.youngs;
	poisson = cmt.poisson;
	shear = cmt.shear;
	mass = LE_Mass->text().toDouble();
	ixx = LE_Ixx->text().toDouble();
	iyy = LE_Iyy->text().toDouble();
	izz = LE_Izz->text().toDouble();
	ixy = LE_Ixy->text().toDouble();
	iyz = LE_Iyz->text().toDouble();
	izx = LE_Izx->text().toDouble();
	QStringList loc = LE_Position->text().split(",");
	x = loc.at(0).toDouble();
	y = loc.at(1).toDouble();
	z = loc.at(2).toDouble();
	this->close();
	this->setResult(QDialog::Accepted);
}

void bodyInfoDialog::Click_cancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}