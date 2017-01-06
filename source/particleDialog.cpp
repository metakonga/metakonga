#include "particleDialog.h"
#include "particle_system.h"
#include "modeler.h"
#include "object.h"
#include "msgBox.h"
#include <QtWidgets>

particleDialog::particleDialog()
{
	LMaterial = new QLabel("Material");
	CBMaterial = new QComboBox;
	CBMaterial->addItems(getMaterialList());
}

particleDialog::~particleDialog()
{

}

particle_system* particleDialog::callDialog(modeler *_md)
{
	md = _md;
	QList<QString> geoList = _md->objects().keys();
	QStringList geoStrList;
	for (unsigned int i = 0; i < geoList.size(); i++)
		geoStrList.push_back(geoList[i]);
	
	tabWidget = new QTabWidget;
	byGeoTab = new QWidget;
	byManualTab = new QWidget;
	QLabel *LBaseGeometry = new QLabel("Base geometry");
	CBGeometry = new QComboBox;
	QLabel *LName = new QLabel("Name");
	LEName = new QLineEdit;
	LEName->setText("particles");
	LEName->setReadOnly(true);
	QLabel *LRadius = new QLabel("Radius");
	QLabel *LMRadius = new QLabel("Radius");
	QLabel *LSpacing = new QLabel("Spacing");
	QLabel *LNumParticle = new QLabel("Num. particle");
	QLabel *LTotalMass = new QLabel("Total mass");
	QLabel *LCohesion = new QLabel("Cohesion");
	LESpacing = new QLineEdit;
	LESpacing->setText("1e-6");
	LERadius = new QLineEdit;
	LEMRadius = new QLineEdit;
	LETotalMass = new QLineEdit;
	LENumParticle = new QLineEdit;
	LECohesion = new QLineEdit;
	QLabel *LRestitution = new QLabel("Restitution");
	LERestitution = new QLineEdit;
	QLabel *LStiffRatio = new QLabel("Stiffness Ratio");
	LEStiffRatio = new QLineEdit;
	QLabel *Lfriction = new QLabel("Friction");
	LEFriction = new QLineEdit;
	QGridLayout *byGeoLayout = new QGridLayout;
	
	CBGeometry->addItems(geoStrList);
	byGeoLayout->addWidget(LBaseGeometry, 0, 0); byGeoLayout->addWidget(CBGeometry, 0, 1, 1, 2);
	byGeoLayout->addWidget(LName, 1, 0); byGeoLayout->addWidget(LEName, 1, 1, 1, 2);
	byGeoLayout->addWidget(LRadius, 2, 0); byGeoLayout->addWidget(LERadius, 2, 1, 1, 2);
	byGeoLayout->addWidget(LSpacing, 3, 0); byGeoLayout->addWidget(LESpacing, 3, 1, 1, 2);
	byGeoLayout->addWidget(LNumParticle, 4, 0); byGeoLayout->addWidget(LENumParticle, 4, 1, 1, 2);
	byGeoLayout->addWidget(LTotalMass, 5, 0); byGeoLayout->addWidget(LETotalMass, 5, 1, 1, 2);
	byGeoLayout->addWidget(LRestitution, 6, 0); byGeoLayout->addWidget(LERestitution, 6, 1, 1, 2);
	byGeoLayout->addWidget(LStiffRatio, 7, 0); byGeoLayout->addWidget(LEStiffRatio, 7, 1, 1, 2);
	byGeoLayout->addWidget(Lfriction, 8, 0); byGeoLayout->addWidget(LEFriction, 8, 1, 1, 2);
	byGeoLayout->addWidget(LCohesion, 9, 0); byGeoLayout->addWidget(LECohesion, 9, 1, 1, 2);
	byGeoTab->setLayout(byGeoLayout);
	tabWidget->addTab(byGeoTab, "Geometry"); byGeoTab->setObjectName("Geometry");
	QLabel *LPosition = new QLabel("Position");
	LEPosition = new QLineEdit;
	QLabel *LVelocity = new QLabel("Velocity");
	LEVelocity = new QLineEdit;

	QGridLayout *byManualLayout = new QGridLayout;
	byManualLayout->addWidget(LMaterial, 0, 0);
	byManualLayout->addWidget(CBMaterial, 0, 1, 1, 2);
	byManualLayout->addWidget(LPosition, 1, 0);
	byManualLayout->addWidget(LEPosition, 1, 1, 1, 2);
	byManualLayout->addWidget(LVelocity, 2, 0);
	byManualLayout->addWidget(LEVelocity, 2, 1, 1, 2);
	byManualLayout->addWidget(LMRadius, 3, 0);
	byManualLayout->addWidget(LEMRadius, 3, 1, 1, 2);
	byManualTab->setLayout(byManualLayout);
	tabWidget->addTab(byManualTab, "Manual"); byManualTab->setObjectName("Manual");
	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(buttonBox, SIGNAL(accepted()), this, SLOT(Click_ok()));
	connect(buttonBox, SIGNAL(rejected()), this, SLOT(Click_cancel()));
	connect(LERadius, SIGNAL(editingFinished()), this, SLOT(particleInformation()));
	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(tabWidget);
	mainLayout->addWidget(buttonBox);
	this->setLayout(mainLayout);

	this->exec();
	particle_system* ps = NULL;
	if (isDialogOk){
		ps = _md->makeParticleSystem(LEName->text());
		object* b_obj = _md->objectFromStr(CBGeometry->currentText());
		if (ps->isMemoryAlloc()){
			ps->addParticles(b_obj);
		}
		else{
			ps->makeParticles(b_obj, LESpacing->text().toFloat(), LERadius->text().toFloat());
			ps->setCollision(LERestitution->text().toFloat(), LEStiffRatio->text().toFloat(), LEFriction->text().toFloat(), LECohesion->text().toFloat());
		}
		
		//ps->makeParticles
	}
	return ps;
}

void particleDialog::Click_ok()
{
	bool isManual;
	float p[4] = { 0, };
	float v[4] = { 0, };
	QWidget *cTab = tabWidget->currentWidget();
	QString on = cTab->objectName();
	if (on == "Geometry"){
		if (LEName->text().isEmpty()){
			msgBox("Value of name is empty!!", QMessageBox::Critical);
			return;
		}
		if (LERadius->text().isEmpty()){
			msgBox("Value of radius is empty!!", QMessageBox::Critical);
			return;
		}
		if (LERestitution->text().isEmpty()){
			msgBox("Value of radius is empty!!", QMessageBox::Critical);
			return;
		}
		if (LEStiffRatio->text().isEmpty()){
			msgBox("Value of radius is empty!!", QMessageBox::Critical);
			return;
		}
		if (LEFriction->text().isEmpty()){
			msgBox("Value of radius is empty!!", QMessageBox::Critical);
			return;
		}
	//	baseGeometry = CBGeometry->currentText();
		//this->setObjectName(LEName->text());// = LEName->text();
		isManual = false;
		//radius = LERadius->text().toFloat();
		// 		QString comm;
		// 		QTextStream(&comm) << "Create Geometry " << baseGeometry << " " << radius << "\n";
	}
	else if (on == "Manual"){
// 		QStringList ss;// = LEPa->text().split(",");
// 		if (LEMRadius->text().isEmpty()){
// 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// 			return;
// 		}
// 		if (LEPosition->text().split(",").size() == 3)
// 			ss = LEPosition->text().split(",");
// 		else
// 			if (LEPosition->text().split(" ").size() == 3)
// 				ss = LEPosition->text().split(" ");
// 			else
// 				if (LEPosition->text().split(", ").size() == 3)
// 					ss = LEPosition->text().split(", ");
// 				else {
// 					msgBox("Position is wrong data.", QMessageBox::Critical);
// 					return;
// 				}
// 				p[0] = ss.at(0).toFloat(); p[1] = ss.at(1).toFloat(); p[2] = ss.at(2).toFloat();
// 
// 				if (LEVelocity->text().split(",").size() == 3)
// 					ss = LEVelocity->text().split(",");
// 				else
// 					if (LEVelocity->text().split(" ").size() == 3)
// 						ss = LEVelocity->text().split(" ");
// 					else
// 						if (LEVelocity->text().split(", ").size() == 3)
// 							ss = LEVelocity->text().split(", ");
// 						else{
// 							msgBox("Velocity is wrong data.", QMessageBox::Critical);
// 							return;
// 						}
// 						v[0] = ss.at(0).toFloat(); v[1] = ss.at(1).toFloat(); v[2] = ss.at(2).toFloat();
// 						isManual = true;
// // 						baseGeometry = "none";
// // 						Object::name = "particles";
// // 						radius = p[3] = LEMRadius->text().toFloat();
// // 						mtype = material_str2enum(CBMaterial->currentText().toStdString());
// // 						material = getMaterialConstant(mtype);
// // 						AddParticleFromManual(p, v);
// 						// 		QString comm;
// 						// 		QTextStream(&comm) 
// 						// 			<< "Create Manual " << radius << " " << (int)mtype << " "
// 						// 			<< p[0] << " " << p[1] << " " << p[2] << " "
// 						// 			<< v[0] << " " << p[1] << " " << p[2] << "\n";
// 						// 		cpProcess.push_back(comm);
	}

	this->close();
	isDialogOk = true;
}

void particleDialog::Click_cancel()
{
	this->close();
	isDialogOk = false;
}

void particleDialog::particleInformation()
{
	object* obj = md->objectFromStr(CBGeometry->currentText());
	float rad = LERadius->text().toFloat();
	unsigned int np = obj->makeParticles(rad, LESpacing->text().toFloat(), true);
	float ms = obj->density() * 4.0f * (float)M_PI * pow(rad, 3.0f) / 3.0f;
	float tms = ms * np;
	QString tx;
	QTextStream(&tx) << np;
	LENumParticle->setText(tx); tx.clear();
	QTextStream(&tx) << tms;
	LETotalMass->setText(tx);
}