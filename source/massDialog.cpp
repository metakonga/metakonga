// #include "massDialog.h"
// #include "mphysics_types.h"
// #include "msgBox.h"
// #include "checkFunctions.h"
// #include "modeler.h"
// #include "object.h"
// #include "mass.h"
// #include <QtWidgets>
// 
// massDialog::massDialog()
// 	: isDialogOk(false)
// {
// 
// }
// 
// massDialog::~massDialog()
// {
// 
// }
// 
// mass* massDialog::callDialog(modeler *md)
// {
// 	QList<QString> geoList = md->objects().keys();
// 	QStringList geoStrList;
// 	for (unsigned int i = 0; i < geoList.size(); i++){
// 		geoStrList.push_back(geoList[i]);
// 	}
// 	QComboBox *CBBase = new QComboBox;
// 	CBBase->addItems(geoStrList);
// 	QLabel *LCBBase = new QLabel("Base geometry");
// 	QLabel *LMass, *LIxx, *LIyy, *LIzz, *LIxy, *LIxz, *LIzy;
// 	QLineEdit *LEMass, *LEIxx, *LEIyy, *LEIzz, *LEIxy, *LEIxz, *LEIzy;
// 	
// 	LMass = new QLabel("Mass"); LEMass = new QLineEdit;
// 	LIxx = new QLabel("Ixx");	LEIxx = new QLineEdit;
// 	LIyy = new QLabel("Iyy");   LEIyy = new QLineEdit;
// 	LIzz = new QLabel("Izz");	LEIzz = new QLineEdit;
// 	LIxy = new QLabel("Ixy");	LEIxy = new QLineEdit;
// 	LIxz = new QLabel("Ixz");	LEIxz = new QLineEdit;
// 	LIzy = new QLabel("Izy");	LEIzy = new QLineEdit;
// 	
// 	PBOk = new QPushButton("OK");
// 	PBCancel = new QPushButton("Cancel");
// 	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
// 	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
// 	QGridLayout *massLayout = new QGridLayout;
// 	massLayout->addWidget(LCBBase, 0, 0);
// 	massLayout->addWidget(CBBase, 0, 1, 1, 2);
// 	massLayout->addWidget(LMass, 1, 0);
// 	massLayout->addWidget(LEMass, 1, 1, 1, 2);
// 	massLayout->addWidget(LIxx, 2, 0);
// 	massLayout->addWidget(LEIxx, 2, 1, 1, 1);
// 	massLayout->addWidget(LIxy, 3, 0);
// 	massLayout->addWidget(LEIxy, 3, 1, 1, 1);
// 	massLayout->addWidget(LIyy, 3, 2);
// 	massLayout->addWidget(LEIyy, 3, 3, 1, 1);
// 	massLayout->addWidget(LIxz, 4, 0);
// 	massLayout->addWidget(LEIxz, 4, 1, 1, 1);
// 	massLayout->addWidget(LIzy, 4, 2);
// 	massLayout->addWidget(LEIzy, 4, 3, 1, 1);
// 	massLayout->addWidget(LIzz, 4, 4);
// 	massLayout->addWidget(LEIzz, 4, 5, 1, 1);
// 	massLayout->addWidget(PBOk, 5, 4);
// 	massLayout->addWidget(PBCancel, 5, 5);
// 	this->setLayout(massLayout);
// 	this->exec();
// 	mass* m = NULL;
// 	if (isDialogOk)
// 	{
// 		m = md->makeMass(CBBase->currentText());
// 		m->setMass(LEMass->text().toDouble());		
// 
// 		m->setBaseGeometryType(md->objects().find(m->name()).value()->objectType());
// 		VEC3D syminer;							//inertia
// 		syminer.x = LEIxy->text().toDouble();
// 		syminer.y = LEIxz->text().toDouble();
// 		syminer.z = LEIzy->text().toDouble();
// 		m->setSymIner(syminer);
// 
// 		VEC3D prininer;
// 		prininer.x = LEIxx->text().toDouble();
// 		prininer.y = LEIyy->text().toDouble();
// 		prininer.z = LEIzz->text().toDouble();
// 		m->setPrinIner(prininer);
// 		m->setInertia();//m->define();
// 	}
// 	return m;
// }
// 
// void massDialog::Click_ok()
// {
// 	this->close();
// 	isDialogOk = true;
// }
// 
// void massDialog::Click_cancel()
// {
// 	this->close();
// 	isDialogOk = false;
// }