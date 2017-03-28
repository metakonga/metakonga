#include "hmcmDialog.h"
#include "modeler.h"
#include "object.h"

hmcmDialog::hmcmDialog(QWidget* parent, modeler *_md)
	: QDialog(parent)
	, md(_md)
{
	setupUi(this);
	setupDialog();
}

hmcmDialog::~hmcmDialog()
{

}

void hmcmDialog::setupDialog()
{
	size_t ncon = md->collisions().size();
	QString name = "collision";
	QTextStream(&name) << ncon;
	LEName->setText(name);
	QList<QString> geoList = md->objects().keys();
	QStringList geoStrList;
	geoStrList.push_back(md->particleSystem()->name());
	for (unsigned int i = 0; i < geoList.size(); i++){
		if (geoList[i] == md->particleSystem()->baseObject())
			continue;
		geoStrList.push_back(geoList[i]);
	}
	CBFirstObject->addItems(geoStrList);
	CBSecondObject->addItems(geoStrList);
	LERestitution->setText("0.8");
	LEFriction->setText("0.3");
	LERollingFriction->setText("0.05");
	LECohesion->setEnabled(false);
	LERatio->setEnabled(false);

	connect(PBOk, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(PBCancle, SIGNAL(clicked()), this, SLOT(click_cancle()));
	connect(CHBEableCohesion, SIGNAL(clicked()), this, SLOT(check_cohesion()));
	connect(CHBUSESTIFFNESSRATIO, SIGNAL(clicked()), this, SLOT(check_stiffnessRatio()));
	exec();
}

void hmcmDialog::click_ok()
{
	tContactModel tcm = HMCM;
	collision *cs = NULL;
	QString ostr1 = CBFirstObject->currentText();
	QString ostr2 = CBSecondObject->currentText();
	object *o1 = NULL;
	object *o2 = NULL;
	particle_system *ps = NULL;
	//particle_system *ps2 = NULL;
	float coh = 0.f;
	float ratio = 0.f;
	if (CHBEableCohesion->isChecked()){
		coh = LECohesion->text().toDouble();
	}
	if (CHBUSESTIFFNESSRATIO->isChecked()){
		ratio = LERatio->text().toDouble();
		tcm = DHS;
	}
	if (ostr1 == "particles" && ostr2 != "particles"){
		ps = md->particleSystem();
		o2 = md->objectFromStr(ostr2);
	}
	else if (ostr2 == "particles" && ostr1 != "particles"){
		o1 = md->objectFromStr(ostr1);
		ps = md->particleSystem();
	}
	else if (ostr1 == "particles" && ostr2 == "particles"){
		ps = md->particleSystem();
	}
	else{
		o1 = md->objectFromStr(ostr1);
		o2 = md->objectFromStr(ostr2);
	}
	tCollisionPair cp = getCollisionPair(o1 ? o1->objectType() : ps->objectType(), o2 ? o2->objectType() : ps->objectType());
	if (!o1 && o2)
		cs = md->makeCollision(LEName->text(), LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), coh, ratio, cp, tcm, ps, o2);
	else if (!o2 && o1)
		cs = md->makeCollision(LEName->text(), LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), coh, ratio, cp, tcm, ps, o1);
	else if (!o1 && !o2)
		cs = md->makeCollision(LEName->text(), LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), coh, ratio, cp, tcm, ps, ps);
	else
		cs = md->makeCollision(LEName->text(), LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), coh, ratio, cp, tcm, o1, o2);

	close();
}

void hmcmDialog::click_cancle()
{
	close();
}

void hmcmDialog::check_cohesion()
{
	bool isChecked = CHBEableCohesion->isChecked();
	if (isChecked)
		LECohesion->setEnabled(true);
	else
		LECohesion->setEnabled(false);
}

void hmcmDialog::check_stiffnessRatio()
{
	bool isChecked = CHBUSESTIFFNESSRATIO->isChecked();
	if (isChecked)
		LERatio->setEnabled(true);
	else
		LERatio->setEnabled(false);
}