#include "ccDialog.h"
#include "modeler.h"
#include "collision.h"
#include "msgBox.h"
#include "object.h"
#include <QtWidgets>

ccDialog::ccDialog()
	: isDialogOk(false)
{

}

ccDialog::~ccDialog()
{

}

collision* ccDialog::callDialog(modeler *md)
{
	//md->objectFromStr(QString("particle"));
	QList<QString> geoList = md->objects().keys();
	QStringList geoStrList;
	geoStrList.push_back(md->particleSystem()->name());
	for (unsigned int i = 0; i < geoList.size(); i++){
		if (geoList[i] == md->particleSystem()->baseObject())
			continue;
		geoStrList.push_back(geoList[i]);
	}

	list1 = new QComboBox; list1->addItems(geoStrList);
	list2 = new QComboBox; list2->addItems(geoStrList);
	QLabel *LName = new QLabel("Name");
	QLabel *LList1 = new QLabel("First Object");
	QLabel *LList2 = new QLabel("Second Object");
	//QLabel *LDamping = new QLabel("Damping")
	QLabel *LCohesion = new QLabel("Cohesion");
	QLabel *Lrest = new QLabel("Restitution");
	QLabel *LRatio = new QLabel("Stiffness ratio");
	QLabel *LFric = new QLabel("Friction");
	LErest = new QLineEdit;
	LEfric = new QLineEdit;
	LECohesion = new QLineEdit;
	LEratio = new QLineEdit;
	LEName = new QLineEdit;

	QString nm;
	QTextStream(&nm) << "collision" << md->numCollision();

	LEName->setText(nm);
	QGridLayout *ccLayout = new QGridLayout;
	QPushButton *PBOk = new QPushButton("OK");
	QPushButton *PBCancel = new QPushButton("Cancel");
	connect(PBOk, SIGNAL(clicked()), this, SLOT(clickOk()));
	connect(PBCancel, SIGNAL(clicked()), this, SLOT(clickCancel()));
	ccLayout->addWidget(LName, 0, 0);	ccLayout->addWidget(LEName, 0, 1, 1, 2);
	ccLayout->addWidget(LList1, 1, 0);	ccLayout->addWidget(list1, 1, 1, 1, 2);
	ccLayout->addWidget(LList2, 2, 0);	ccLayout->addWidget(list2, 2, 1, 1, 2);
	ccLayout->addWidget(Lrest, 3, 0);	ccLayout->addWidget(LErest, 3, 1, 1, 2);
	ccLayout->addWidget(LRatio, 4, 0);	ccLayout->addWidget(LEratio, 4, 1, 1, 2);
	ccLayout->addWidget(LFric, 5, 0);	ccLayout->addWidget(LEfric, 5, 1, 1, 2);
	ccLayout->addWidget(LCohesion, 6, 0);  ccLayout->addWidget(LECohesion, 6, 1, 1, 2);
	ccLayout->addWidget(PBOk, 7, 0);	ccLayout->addWidget(PBCancel, 7, 1);
	this->setLayout(ccLayout);

	this->exec();

	collision *cs = NULL;
	if (isDialogOk)
	{
		QString ostr1 = list1->currentText();
		QString ostr2 = list2->currentText();
		object *o1 = NULL;
		object *o2 = NULL;
		particle_system *ps = NULL;
		
		if (ostr1 == "particles"){
			ps = md->particleSystem();
			o2 = md->objectFromStr(ostr2);
		}
		else if(ostr2 == "particles"){
			o1 = md->objectFromStr(ostr1);
			ps = md->particleSystem();
		}
		else{
			o1 = md->objectFromStr(ostr1);
			o2 = md->objectFromStr(ostr2);
		}
		tCollisionPair cp = getCollisionPair(o1 ? o1->objectType() : ps->objectType(), o2 ? o2->objectType() : ps->objectType());
		if(!o1)
			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEratio->text().toFloat(), LEfric->text().toFloat(), LECohesion->text().toFloat(), cp, ps, o2);
		else if (!o2)
			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEratio->text().toFloat(), LEfric->text().toFloat(), LECohesion->text().toFloat(), cp, ps, o1);
		else
			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEratio->text().toFloat(), LEfric->text().toFloat(), LECohesion->text().toFloat(), cp, o1, o2);
	}

	return cs;
}

void ccDialog::clickOk()
{
	if (list1->currentText() == list2->currentText()){
		msgBox("You selected same object.", QMessageBox::Critical);
		return;
	}

	if (LErest->text().isEmpty() || LEfric->text().isEmpty() || LEratio->text().isEmpty()){
		msgBox("There is empty input.", QMessageBox::Critical);
		return;
	}

	this->close();
	isDialogOk = true;
}

void ccDialog::clickCancel()
{
	this->close();
	isDialogOk = false;
}