// #include "polygonDialog.h"
// #include "mphysics_types.h"
// #include "msgBox.h"
// #include "checkFunctions.h"
// #include "modeler.h"
// #include "polygon.h"
// #include <QtWidgets>
// 
// polygonDialog::polygonDialog()
// 	: isDialogOk(false)
// {
// 	
// }
// 
// polygonDialog::~polygonDialog()
// {
// 
// }
// 
// polygon* polygonDialog::callDialog(modeler *md)
// {
// 	QString nm;
// 	QTextStream(&nm) << "polygon" << md->numPolygon();
// 	this->setObjectName("Polygon Dialog");
// 	QLabel *LName = new QLabel("Name");
// 	QLabel *LP = new QLabel("point P");
// 	QLabel *LQ = new QLabel("point Q");
// 	QLabel *LR = new QLabel("point R");
// 	QLabel *LMaterial = new QLabel("Material");
// 	LEName = new QLineEdit;
// 	LEName->setText(nm);
// 	LEP = new QLineEdit;
// 	LEQ = new QLineEdit;
// 	LER = new QLineEdit;
// 	CBMaterial = new QComboBox;
// 	CBMaterial->addItems(getMaterialList());
// 	PBOk = new QPushButton("OK");
// 	PBCancel = new QPushButton("Cancel");
// 	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
// 	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
// 	polyLayout = new QGridLayout;
// 	polyLayout->addWidget(LMaterial, 0, 0);
// 	polyLayout->addWidget(CBMaterial, 0, 1, 1, 2);
// 	polyLayout->addWidget(LName, 1, 0);
// 	polyLayout->addWidget(LEName, 1, 1, 1, 2);
// 	polyLayout->addWidget(LP, 2, 0);
// 	polyLayout->addWidget(LEP, 2, 1, 1, 2);
// 	polyLayout->addWidget(LQ, 3, 0);
// 	polyLayout->addWidget(LEQ, 3, 1, 1, 2);
// 	polyLayout->addWidget(LR, 4, 0);
// 	polyLayout->addWidget(LER, 4, 1, 1, 2);
// 	polyLayout->addWidget(PBOk, 5, 0);
// 	polyLayout->addWidget(PBCancel, 5, 1);
// 	this->setLayout(polyLayout);
// 
// 	this->exec();
// 	polygon *po = NULL;
// 	if (isDialogOk)
// 	{
// 		po = md->makePolygon(this->objectName(), tMaterial(CBMaterial->currentIndex()), ROLL_BOUNDARY);
// 		QStringList chList = LEP->text().split(" ");
// 		float p[3] = { chList.at(0).toFloat(), chList.at(1).toFloat(), chList.at(2).toFloat() };
// 		chList = LEQ->text().split(" ");
// 		float q[3] = { chList.at(0).toFloat(), chList.at(1).toFloat(), chList.at(2).toFloat() };
// 		chList = LER->text().split(" ");
// 		float r[3] = { chList.at(0).toFloat(), chList.at(1).toFloat(), chList.at(2).toFloat() };
// 		po->define(VEC3F(p), VEC3F(q), VEC3F(r));
// 	}
// 	return po;
// }
// 
// void polygonDialog::Click_ok()
// {
// 	this->close();
// 	isDialogOk = true;
// }
// 
// void polygonDialog::Click_cancel()
// {
// 	this->close();
// 	isDialogOk = false;
// }