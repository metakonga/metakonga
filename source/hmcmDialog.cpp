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

	connect(PBOk, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(PBOk, SIGNAL(clicked()), this, SLOT(click_cancle()));

	exec();
}

void hmcmDialog::click_ok()
{
	collision *cs = NULL;
	QString ostr1 = CBFirstObject->currentText();
	QString ostr2 = CBSecondObject->currentText();
	object *o1 = NULL;
	object *o2 = NULL;
	particle_system *ps = NULL;
	if (ostr1 == "particles"){
		ps = md->particleSystem();
		o2 = md->objectFromStr(ostr2);
	}
	else if (ostr2 == "particles"){
		o1 = md->objectFromStr(ostr1);
		ps = md->particleSystem();
	}
	else{
		o1 = md->objectFromStr(ostr1);
		o2 = md->objectFromStr(ostr2);
	}
	tCollisionPair cp = getCollisionPair(o1 ? o1->objectType() : ps->objectType(), o2 ? o2->objectType() : ps->objectType());
	if (!o1)
		cs = md->makeCollision(LEName->text(), LERestitution->text().toFloat(), LEFriction->text().toFloat(), LERollingFriction->text().toFloat(), cp, HMCM, ps, o2);
	else if (!o2)
		cs = md->makeCollision(LEName->text(), LERestitution->text().toFloat(), LEFriction->text().toFloat(), LERollingFriction->text().toFloat(), cp, HMCM, ps, o1);
	else
		cs = md->makeCollision(LEName->text(), LERestitution->text().toFloat(), LEFriction->text().toFloat(), LERollingFriction->text().toFloat(), cp, HMCM, o1, o2);

	if (CHBEableCohesion->isChecked()){

	}
	close();
}

void hmcmDialog::click_cancle()
{
	close();
}