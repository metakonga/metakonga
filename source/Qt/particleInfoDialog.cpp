#include "particleInfoDialog.h"
#include "ui_particleInfoDialog.h"
#include <QFile>

particleInfoDialog::particleInfoDialog(QWidget *parent) :
QDialog(parent),
ui(new Ui::PInfoDialog)
{
	ui->setupUi(this);
	connect(ui->PHSlider, SIGNAL(valueChanged(int)), this, SLOT(sliderSlot()));
	connect(ui->LEPid, SIGNAL(editingFinished()), this, SLOT(pidLineEditSlot()));
	connect(ui->BChange, SIGNAL(clicked()), this, SLOT(buttonSlot()));
}

particleInfoDialog::~particleInfoDialog()
{
	delete ui;
}

// void particleInfoDialog::bindingParticleViewer(parview::Object *par)
// {
// // 	parsys = dynamic_cast<parview::particles*>(par);
// // 	QString tp;
// // 	tp.sprintf("%d", parsys->Np());
// // 	ui->LEPid->setText("0");
// // 	ui->LETotalParticle->setText(tp);
// // 	ui->PHSlider->setMaximum(parsys->Np());
// // 	ui->PHSlider->setSingleStep(1);
// // 	ui->PHSlider->setPageStep(1);
// }

void particleInfoDialog::updateParticleInfo(unsigned int id)
{
// 	if (id == 0)
// 		return;
// 	vector4<float> pos = parsys->getPositionToV4<float>(id-1);
// 	QString v3str;
// 	v3str.sprintf("%05f, %05f, %05f", pos.x, pos.y, pos.z);
// 	ui->LEPosition->setText(v3str);
// 	
// 	v3str.clear();
// 	vector4<float> vel = parsys->getVelocityToV4<float>(id - 1);
// 	v3str.sprintf("%05f, %05f, %05f", vel.x, vel.y, vel.z);
// 	ui->LEVelocity->setText(v3str);
// 
// 	double pressure = parsys->getPressure(id-1);
// 	QString str;
// 	str.sprintf("%f", pressure);
// 	ui->LEPressure->setText(str);
// 
// 	double fsval = parsys->getFreeSurfaceValue(id - 1);
// 	str.clear();
// 	str.sprintf("%f", fsval);
// 	ui->LEFreeSurfaceValue->setText(str);
// 
// 	bool isfs = parsys->isFreeSurface(id-1);
// 	QString boolStr;
// 	//isfs ? ui->LEFreeSurfac
// 	
// 	parsys->changeParticleColor(id-1);
}

void particleInfoDialog::sliderSlot()
{
	int id = ui->PHSlider->value();
	QString sid;
	sid.sprintf("%d", id);
	ui->LEPid->setText(sid);

	updateParticleInfo(id);
}

void particleInfoDialog::pidLineEditSlot()
{
	QString str = ui->LEPid->text();
	updateParticleInfo(str.toUInt());
	ui->PHSlider->setValue(str.toInt());
}

void particleInfoDialog::buttonSlot()
{
	int id = ui->PHSlider->value();
	//parsys->changeParticleColor(id - 1);
	//parsys->drawSupportSphere(id - 1);
}