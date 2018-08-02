#include "contactPairDialog.h"
#include "contact.h"
#include "messageBox.h"

contactPairDialog::contactPairDialog(QWidget* parent)
	: QDialog(parent)
	, restitution(0.9)
	, stiffnessRatio(0.8)
	, friction(0.3)
{
	setupUi(this);
	name = "CollisionPair" + QString("%1").arg(contact::count);
	LE_Name->setText(name);
	LE_Restitution->setText(QString("%1").arg(restitution));
	LE_StiffnessRatio->setText(QString("%1").arg(stiffnessRatio));
	LE_Friction->setText(QString("%1").arg(friction));

	connect(CB_FirstObject, SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBox(int)));
	connect(CB_SecondObject, SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBox(int)));
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(click_cancle()));
}

contactPairDialog::~contactPairDialog()
{

}

void contactPairDialog::changeComboBox(int idx)
{
	QComboBox *cb = dynamic_cast<QComboBox*>(sender());
	QString fs = CB_FirstObject->currentText();
	QString ss = CB_SecondObject->currentText();
	if (fs != "particles" && ss != "particles")
	{
		messageBox::run("Selected pair is not supported in this version.");
		cb->setCurrentIndex(0);
	}
}

void contactPairDialog::click_ok()
{
	method = TB_METHOD->currentIndex();
	name = LE_Name->text();
	firstObj = CB_FirstObject->currentText();
	secondObj = CB_SecondObject->currentText();
	restitution = LE_Restitution->text().toDouble();
	stiffnessRatio = LE_StiffnessRatio->text().toDouble();
	friction = LE_Friction->text().toDouble();
	this->close();
	this->setResult(QDialog::Accepted);
}

void contactPairDialog::click_cancle()
{
	this->close();
	this->setResult(QDialog::Rejected);
}

void contactPairDialog::setObjectLists(QStringList& list)
{
	CB_FirstObject->addItems(list);
	CB_SecondObject->addItems(list);
}

// #include "ccDialog.h"
// #include "modeler.h"
// #include "collision.h"
// #include "msgBox.h"
// #include "object.h"
// #include <QtWidgets>
// 
// ccDialog::ccDialog()
// 	: isDialogOk(false)
// {
// 
// }
// 
// ccDialog::~ccDialog()
// {
// 
// }
// 
// collision* ccDialog::callDialog(modeler *md)
// {
// 	//md->objectFromStr(QString("particle"));
// 	QList<QString> geoList = md->objects().keys();
// 	QStringList geoStrList;
// 	geoStrList.push_back(md->particleSystem()->name());
// 	for (unsigned int i = 0; i < geoList.size(); i++){
// 		if (geoList[i] == md->particleSystem()->baseObject())
// 			continue;
// 		geoStrList.push_back(geoList[i]);
// 	}
// 
// 	list1 = new QComboBox; list1->addItems(geoStrList);
// 	list2 = new QComboBox; list2->addItems(geoStrList);
// 	QLabel *LName = new QLabel("Name");
// 	QLabel *LList1 = new QLabel("First Object");
// 	QLabel *LList2 = new QLabel("Second Object");
// 	//QLabel *LDamping = new QLabel("Damping")
// 	QLabel *LCohesion = new QLabel("Cohesion");
// 	QLabel *Lrest = new QLabel("Restitution");
// 	QLabel *LRollingFriction = new QLabel("Rolling friction");
// 	QLabel *LFric = new QLabel("Friction");
// 	LErest = new QLineEdit;
// 	LEfric = new QLineEdit;
// 	LECohesion = new QLineEdit;
// 	LERollingFriction = new QLineEdit;
// 	LEName = new QLineEdit;
// 
// 	QString nm;
// 	QTextStream(&nm) << "collision" << md->numCollision();
// 
// 	LEName->setText(nm);
// 	QGridLayout *ccLayout = new QGridLayout;
// 	QPushButton *PBOk = new QPushButton("OK");
// 	QPushButton *PBCancel = new QPushButton("Cancel");
// 	connect(PBOk, SIGNAL(clicked()), this, SLOT(clickOk()));
// 	connect(PBCancel, SIGNAL(clicked()), this, SLOT(clickCancel()));
// 	ccLayout->addWidget(LName, 0, 0);	ccLayout->addWidget(LEName, 0, 1, 1, 2);
// 	ccLayout->addWidget(LList1, 1, 0);	ccLayout->addWidget(list1, 1, 1, 1, 2);
// 	ccLayout->addWidget(LList2, 2, 0);	ccLayout->addWidget(list2, 2, 1, 1, 2);
// 	ccLayout->addWidget(Lrest, 3, 0);	ccLayout->addWidget(LErest, 3, 1, 1, 2);
// 	//ccLayout->addWidget(LShearModulus, 4, 0);	ccLayout->addWidget(LEShear, 4, 1, 1, 2);
// 	ccLayout->addWidget(LFric, 4, 0);	ccLayout->addWidget(LEfric, 4, 1, 1, 2);
// 	ccLayout->addWidget(LRollingFriction, 5, 0); ccLayout->addWidget(LERollingFriction, 5, 1, 1, 2);
// 	ccLayout->addWidget(LCohesion, 6, 0);  ccLayout->addWidget(LECohesion, 6, 1, 1, 2);
// 	ccLayout->addWidget(PBOk, 7, 0);	ccLayout->addWidget(PBCancel, 7, 1);
// 	this->setLayout(ccLayout);
// 
// 	this->exec();
// 
// // 	collision *cs = NULL;
// // 	if (isDialogOk)
// // 	{
// // 		QString ostr1 = list1->currentText();
// // 		QString ostr2 = list2->currentText();
// // 		object *o1 = NULL;
// // 		object *o2 = NULL;
// // 		particle_system *ps = NULL;
// // 		
// // 		if (ostr1 == "particles"){
// // 			ps = md->particleSystem();
// // 			o2 = md->objectFromStr(ostr2);
// // 		}
// // 		else if(ostr2 == "particles"){
// // 			o1 = md->objectFromStr(ostr1);
// // 			ps = md->particleSystem();
// // 		}
// // 		else{
// // 			o1 = md->objectFromStr(ostr1);
// // 			o2 = md->objectFromStr(ostr2);
// // 		}
// // 		tCollisionPair cp = getCollisionPair(o1 ? o1->objectType() : ps->objectType(), o2 ? o2->objectType() : ps->objectType());
// // 		if(!o1)
// // 			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEfric->text().toFloat(), LERollingFriction->text().toFloat(), LECohesion->text().toFloat(), cp, ps, o2);
// // 		else if (!o2)
// // 			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEfric->text().toFloat(), LERollingFriction->text().toFloat(), LECohesion->text().toFloat(), cp, ps, o1);
// // 		else
// // 			cs = md->makeCollision(LEName->text(), LErest->text().toFloat(), LEfric->text().toFloat(), LERollingFriction->text().toFloat(), LECohesion->text().toFloat(), cp, o1, o2);
// // 	}
// 
// 	return 0;// cs;
// }
// 
// void ccDialog::clickOk()
// {
// 	if (list1->currentText() == list2->currentText()){
// 		msgBox("You selected same object.", QMessageBox::Critical);
// 		return;
// 	}
// 
// 	if (LErest->text().isEmpty() || LEfric->text().isEmpty() || LERollingFriction->text().isEmpty()){
// 		msgBox("There is empty input.", QMessageBox::Critical);
// 		return;
// 	}
// 
// 	this->close();
// 	isDialogOk = true;
// }
// 
// void ccDialog::clickCancel()
// {
// 	this->close();
// 	isDialogOk = false;
// }