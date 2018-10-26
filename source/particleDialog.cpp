#include "particleDialog.h"
#include "particleManager.h"
#include "types.h"

particleDialog::particleDialog(QWidget* parent)
	: QDialog(parent)
	, spacing(0.001)
	, min_radius(0.01)
	, max_radius(0.01)
	, circle_diameter(0.01)
	, ncubex(3)
	, ncubey(3)
	, ncubez(3)
	, nplanex(3)
	, nplanez(3)
	, np(0)
	, perNp(0)
	, real_time(false)
{
	setupUi(this);
	LE_NUM_PARTICLE->setReadOnly(false);
	dir[0] = 0.0; dir[1] = 1.0; dir[2] = 0.0;
	loc[0] = 0.0; loc[1] = 0.02; loc[2] = 0.0;
	name = "Particle" + QString("%1").arg(particleManager::count);
	setCubeData();
	LE_Name->setText(name);
	CB_Type->addItems(getMaterialList());
	int mt = CB_Type->currentIndex();
	cmaterialType cmt = getMaterialConstant(mt);
	LE_Youngs->setText(QString("%1").arg(cmt.youngs));
	LE_PoissonRatio->setText(QString("%1").arg(cmt.poisson));
	LE_Density->setText(QString("%1").arg(cmt.density));
	LE_ShearModulus->setText(QString("%1").arg(cmt.shear));
	LE_PLANE_DIR->setText("0.0 0.0 0.0");
	TB_Method->setCurrentIndex(0);
	connect(TB_Method, SIGNAL(currentChanged(int)), this, SLOT(changeTab(int)));
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(click_cancle()));
	connect(CB_Type, SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBox(int)));
	connect(LE_CUBE_NX, SIGNAL(editingFinished()), this, SLOT(update_tnp()));
	connect(LE_CUBE_NY, SIGNAL(editingFinished()), this, SLOT(update_tnp()));
	connect(LE_CUBE_NZ, SIGNAL(editingFinished()), this, SLOT(update_tnp()));
	connect(LE_PLANE_NX, SIGNAL(editingFinished()), this, SLOT(update_tnp()));
	connect(LE_PLANE_NZ, SIGNAL(editingFinished()), this, SLOT(update_tnp()));
}

particleDialog::~particleDialog()
{

}

void particleDialog::changeComboBox(int idx)
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

void particleDialog::setCubeData()
{
	LE_CUBE_NX->setText(QString("%1").arg(ncubex));
	LE_CUBE_NY->setText(QString("%1").arg(ncubey));
	LE_CUBE_NZ->setText(QString("%1").arg(ncubez));
	LE_P_SPACING->setText(QString("%1").arg(spacing));
	LE_MIN_RADIUS->setText(QString("%1").arg(min_radius));
	LE_MAX_RADIUS->setText(QString("%1").arg(max_radius));
	LE_NUM_PARTICLE->setText(QString("%1").arg(ncubex * ncubey * ncubez));
	LE_CUBE_LOC->setText("0.0 0.02 0.0");
}

void particleDialog::setPlaneData()
{
	LE_PLANE_NX->setText(QString("%1").arg(nplanex));
	LE_PLANE_NZ->setText(QString("%1").arg(nplanez));
	LE_P_SPACING->setText(QString("%1").arg(spacing));
	LE_MIN_RADIUS->setText(QString("%1").arg(min_radius));
	LE_MAX_RADIUS->setText(QString("%1").arg(max_radius));
	LE_NUM_PARTICLE->setText(QString("%1").arg(nplanex * nplanez));
	LE_PLANE_LOC->setText("0.0 0.02 0.0");
}

void particleDialog::setCircleData()
{
	LE_NumParticle->setText(QString("%1").arg(1));
	LE_CircleDiameter->setText(QString("%1").arg(circle_diameter));
	LE_CircleLocation->setText(QString("%1 %2 %3").arg(loc[0]).arg(loc[1]).arg(loc[2]));
	LE_CircleDirection->setText(QString("%1 %2 %3").arg(dir[0]).arg(dir[1]).arg(dir[2]));
}

void particleDialog::changeTab(int idx)
{
	if (idx == 0)
		setCubeData();
	else if (idx == 1)
		setPlaneData();
	else if (idx == 2)
		setCircleData();
}

void particleDialog::click_ok()
{
	name = LE_Name->text();
	method = TB_Method->currentIndex();
	QStringList dl;
	if (method == 0)
	{
		ncubex = LE_CUBE_NX->text().toUInt();
		ncubey = LE_CUBE_NY->text().toUInt();
		ncubez = LE_CUBE_NZ->text().toUInt();
		dl = LE_CUBE_LOC->text().split(" ");
		loc[0] = dl.at(0).toDouble();
		loc[1] = dl.at(1).toDouble();
		loc[2] = dl.at(2).toDouble();
	}
	else if (method == 1)
	{
		nplanex = LE_PLANE_NX->text().toUInt();
		nplanez = LE_PLANE_NZ->text().toUInt();
		QStringList dl = LE_PLANE_DIR->text().split(" ");
		dir[0] = dl.at(0).toDouble(); 
		dir[1] = dl.at(1).toDouble(); 
		dir[2] = dl.at(2).toDouble();
		dl = LE_CUBE_LOC->text().split(" ");
		loc[0] = dl.at(0).toDouble();
		loc[1] = dl.at(1).toDouble();
		loc[2] = dl.at(2).toDouble();
		np = LE_NUM_PARTICLE->text().toUInt();
	}
	else if (method == 2)
	{
		circle_diameter = LE_CircleDiameter->text().toDouble();
		
		QStringList cl = LE_CircleDirection->text().split(" ");
		dir[0] = cl.at(0).toDouble();
		dir[1] = cl.at(1).toDouble();
		dir[2] = cl.at(2).toDouble();
		cl = LE_CircleLocation->text().split(" ");
		loc[0] = cl.at(0).toDouble();
		loc[1] = cl.at(1).toDouble();
		loc[2] = cl.at(2).toDouble();
		np = LE_NumParticle->text().toUInt();
	}
	spacing = LE_P_SPACING->text().toDouble();
	min_radius = LE_MIN_RADIUS->text().toDouble();
	max_radius = LE_MAX_RADIUS->text().toDouble();

	type = CB_Type->currentIndex();
	youngs = LE_Youngs->text().toDouble();
	poisson = LE_PoissonRatio->text().toDouble();
	density = LE_Density->text().toDouble();
	shear = LE_ShearModulus->text().toDouble();
	real_time = GB_RealTime->isChecked();
	if (real_time)
	{
		one_by_one = RB_OneByOne->isChecked();
		if (one_by_one)
			perNp = LE_NumParclesPer->text().toUInt();
		else
			perTime = LE_NumParclesPer->text().toDouble();
		
	}
		
	this->close();
	this->setResult(QDialog::Accepted);
}

void particleDialog::click_cancle()
{
	this->close();
	this->setResult(QDialog::Rejected);
}

void particleDialog::update_tnp()
{
	method = TB_Method->currentIndex();
	unsigned int np = 0;
	if (method == 0)
	{
		np = LE_CUBE_NX->text().toUInt() * LE_CUBE_NY->text().toUInt() * LE_CUBE_NZ->text().toUInt();
		
	}
	else if (method == 1)
	{
		np = LE_PLANE_NX->text().toUInt() * LE_PLANE_NZ->text().toUInt();
		LE_NUM_PARTICLE->setText(QString("%1").arg(np));
	}
	LE_NUM_PARTICLE->setText(QString("%1").arg(np));
}

void particleDialog::click_radio_button()
{
	if (RB_OneByOne->isChecked())
		LE_NumParclesPer->setText("The number of particles(per second)");
	else
		LE_NumParclesPer->setText("The creating time of(per group)");
}

// #include "particleDialog.h"
// #include "particle_system.h"
// #include "modeler.h"
// #include "object.h"
// #include "msgBox.h"
// #include <QtWidgets>
// 
// particleDialog::particleDialog(QWidget* parent, modeler *_md)
// 	: QDialog(parent)
// 	, md(_md)
// {
// 	setupUi(this);
// 	setupDialog();
// }
// 
// particleDialog::~particleDialog()
// {
// 
// }
// 
// void particleDialog::setupDialog()
// {
// 	//size_t ncon = md->collisions().size();
// 	QString name = "particles";
// 	//QTextStream(&name) << ncon;
// 	//LEName->setText(name);
// 	QList<QString> geoList = md->objects().keys();
// 	QStringList geoStrList;
// //	geoStrList.push_back(md->particleSystem()->name());
// 	for (unsigned int i = 0; i < geoList.size(); i++){
// // 		if (geoList[i] == md->particleSystem()->baseObject())
// // 			continue;
// 		geoStrList.push_back(geoList[i]);
// 	}
// 	CB_BaseGeometry->addItems(geoStrList);
// 	LE_StackNumber->setEnabled(false);
// 	LE_NumParticle->setText("0");
// 	LE_TotalMass->setText("0");
// 	LE_StackNumber->setText("0");
// 
// 	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(click_ok()));
// 	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(click_cancle()));
// 	connect(LE_Radius, SIGNAL(editingFinished()), this, SLOT(change_particle_radius()));
// 	connect(LE_StackNumber, SIGNAL(editingFinished()), this, SLOT(change_stack_number()));
// 	connect(CHB_StackParticle, SIGNAL(clicked()), this, SLOT(check_stack_particle()));
// 	exec();
// 
// 	//CB->addItems(geoStrList);
// }
// 
// void particleDialog::check_stack_particle()
// {
// 	bool isChecked = CHB_StackParticle->isChecked();
// 	if (isChecked)
// 		LE_StackNumber->setEnabled(true);
// 	else
// 		LE_StackNumber->setEnabled(false);
// }
// 
// void particleDialog::change_particle_radius()
// {
// 	QString str;
// 	VEC3D space = 0.f;
// 	VEC3UI size;
// 	object* obj = md->objectFromStr(CB_BaseGeometry->currentText());
// 	float rad = LE_Radius->text().toDouble();
// 	QTextStream(&str) << rad * 0.1f;
// 	LE_Spacing->setText(str);
// 	unsigned int np = obj->makeParticles(rad, size, space, 0, true);
// 	str.clear();
// 	QTextStream(&str) << np;
// 	LE_NumParticle->setText(str);
// 	float ms = obj->density() * 4.0 * M_PI * pow(rad, 3.0) / 3.0;
// 	str.clear();
// 	QTextStream(&str) << ms * np;
// 	LE_TotalMass->setText(str);
// 	str.clear();
// 	QTextStream(&str) << space.x << " " << space.y << " " << space.z;
// 	LE_Spacing->setText(str);
// 	str.clear();
// 	QTextStream(&str) << size.x << " " << size.y << " " << size.z;
// 	LE_Size->setText(str);
// }
// 
// void particleDialog::change_stack_number()
// {
// 	unsigned int np = LE_NumParticle->text().toDouble();
// 	float ms = LE_TotalMass->text().toDouble() / (float)np;
// 	unsigned int t_np = np + np * LE_StackNumber->text().toUInt();
// 	QString str;
// 	QTextStream(&str) << t_np;
// 	LE_NumParticle->setText(str);
// 	str.clear();
// 	QTextStream(&str) << ms * t_np;
// 	LE_TotalMass->setText(str);
// }
// 
// void particleDialog::click_ok()
// {
// 	particle_system* ps = NULL;
// 
// 	ps = md->makeParticleSystem("particles");
// 	object* b_obj = md->objectFromStr(CB_BaseGeometry->currentText());
// 	QStringList slist;// = LE_Spacing->text().split(" ");
// 	slist = LE_Size->text().split(" ");
// 	VEC3UI size = VEC3UI(slist.at(0).toUInt(), slist.at(1).toUInt(), slist.at(2).toUInt());
// 	unsigned int num_stack = 0;
// 	float stack_dt = 0.f;
// 	if (CHB_StackParticle->isChecked()){
// 		num_stack = LE_StackNumber->text().toUInt();
// 		stack_dt = LE_StackTimeInterval->text().toDouble();
// 		ps->setGenerationMethod(STACK_PARTICLE_METHOD, num_stack, stack_dt, size.x*size.y*size.z);
// 		// 		ps->resizeMemoryForStack(LE_NumParticle->text().toUInt());
// 	}
// // 	if (ps->isMemoryAlloc()){
// // 		ps->addParticles(b_obj, size);
// // 	}
// 	else{
// 		slist = LE_Spacing->text().split(" ");
// 		VEC3D space = VEC3D(slist.at(0).toDouble(), slist.at(1).toDouble(), slist.at(2).toDouble());
// // 		slist = LE_Size->text().split(" ");
// // 		VEC3UI size = VEC3UI(slist.at(0).toUInt(), slist.at(1).toUInt(), slist.at(2).toUInt());
// 		ps->makeParticles(b_obj, size, space, LE_Radius->text().toDouble(), num_stack);
// 		//ps->setCollision(LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), LECohesion->text().toDouble(), 0.8f);
// 	}
// 	/*if (CHB_StackParticle->En)*/
// 	
// 	this->close();
// }
// 
// void particleDialog::click_cancle()
// {
// 	this->close();
// }
// 
// // #include "particleDialog.h"
// // #include "particle_system.h"
// // #include "modeler.h"
// // #include "object.h"
// // #include "msgBox.h"
// // #include <QtWidgets>
// // 
// // particleDialog::particleDialog()
// // {
// // 	LMaterial = new QLabel("Material");
// // 	CBMaterial = new QComboBox;
// // 	CBMaterial->addItems(getMaterialList());
// // }
// // 
// // particleDialog::~particleDialog()
// // {
// // 
// // }
// // 
// // particle_system* particleDialog::callDialog(modeler *_md)
// // {
// // 	md = _md;
// // 	QList<QString> geoList = _md->objects().keys();
// // 	QStringList geoStrList;
// // 	for (unsigned int i = 0; i < geoList.size(); i++)
// // 		geoStrList.push_back(geoList[i]);
// // 	
// // 	tabWidget = new QTabWidget;
// // 	byGeoTab = new QWidget;
// // 	byManualTab = new QWidget;
// // 	QLabel *LBaseGeometry = new QLabel("Base geometry");
// // 	CBGeometry = new QComboBox;
// // 	QLabel *LName = new QLabel("Name");
// // 	LEName = new QLineEdit;
// // 	LEName->setText("particles");
// // 	LEName->setReadOnly(true);
// // 	QLabel *LRadius = new QLabel("Radius");
// // 	QLabel *LMRadius = new QLabel("Radius");
// // 	QLabel *LSpacing = new QLabel("Spacing");
// // 	QLabel *LNumParticle = new QLabel("Num. particle");
// // 	QLabel *LTotalMass = new QLabel("Total mass");
// // 	QLabel *LCohesion = new QLabel("Cohesion");
// // 	LESpacing = new QLineEdit;
// // 	LESpacing->setText("1e-6");
// // 	LERadius = new QLineEdit;
// // 	LEMRadius = new QLineEdit;
// // 	LETotalMass = new QLineEdit;
// // 	LENumParticle = new QLineEdit;
// // 	LECohesion = new QLineEdit;
// // 	QLabel *LRestitution = new QLabel("Restitution");
// // 	LERestitution = new QLineEdit;
// // 	//QLabel *LShearModulus = new QLabel("Shear modulus");
// // 	//LEShearModulus = new QLineEdit;
// // 	QLabel *Lfriction = new QLabel("Friction");
// // 	LEFriction = new QLineEdit;
// // 	QLabel *LRollingFriction = new QLabel("Rolling friction");
// // 	LERollingFriction = new QLineEdit;
// // 	QGridLayout *byGeoLayout = new QGridLayout;
// // 	
// // 	CBGeometry->addItems(geoStrList);
// // 	byGeoLayout->addWidget(LBaseGeometry, 0, 0); byGeoLayout->addWidget(CBGeometry, 0, 1, 1, 2);
// // 	byGeoLayout->addWidget(LName, 1, 0); byGeoLayout->addWidget(LEName, 1, 1, 1, 2);
// // 	byGeoLayout->addWidget(LRadius, 2, 0); byGeoLayout->addWidget(LERadius, 2, 1, 1, 2);
// // 	byGeoLayout->addWidget(LSpacing, 3, 0); byGeoLayout->addWidget(LESpacing, 3, 1, 1, 2);
// // 	byGeoLayout->addWidget(LNumParticle, 4, 0); byGeoLayout->addWidget(LENumParticle, 4, 1, 1, 2);
// // 	byGeoLayout->addWidget(LTotalMass, 5, 0); byGeoLayout->addWidget(LETotalMass, 5, 1, 1, 2);
// // 	byGeoLayout->addWidget(LRestitution, 6, 0); byGeoLayout->addWidget(LERestitution, 6, 1, 1, 2);
// // 	//byGeoLayout->addWidget(LShearModulus, 7, 0); byGeoLayout->addWidget(LEShearModulus, 7, 1, 1, 2);
// // 	byGeoLayout->addWidget(Lfriction, 7, 0); byGeoLayout->addWidget(LEFriction, 7, 1, 1, 2);
// // 	byGeoLayout->addWidget(LRollingFriction, 8, 0); byGeoLayout->addWidget(LERollingFriction, 8, 1, 1, 2);
// // 	byGeoLayout->addWidget(LCohesion, 9, 0); byGeoLayout->addWidget(LECohesion, 9, 1, 1, 2);
// // 	byGeoTab->setLayout(byGeoLayout);
// // 	tabWidget->addTab(byGeoTab, "Geometry"); byGeoTab->setObjectName("Geometry");
// // 	QLabel *LPosition = new QLabel("Position");
// // 	LEPosition = new QLineEdit;
// // 	QLabel *LVelocity = new QLabel("Velocity");
// // 	LEVelocity = new QLineEdit;
// // 
// // 	QGridLayout *byManualLayout = new QGridLayout;
// // 	byManualLayout->addWidget(LMaterial, 0, 0);
// // 	byManualLayout->addWidget(CBMaterial, 0, 1, 1, 2);
// // 	byManualLayout->addWidget(LPosition, 1, 0);
// // 	byManualLayout->addWidget(LEPosition, 1, 1, 1, 2);
// // 	byManualLayout->addWidget(LVelocity, 2, 0);
// // 	byManualLayout->addWidget(LEVelocity, 2, 1, 1, 2);
// // 	byManualLayout->addWidget(LMRadius, 3, 0);
// // 	byManualLayout->addWidget(LEMRadius, 3, 1, 1, 2);
// // 	byManualTab->setLayout(byManualLayout);
// // 	tabWidget->addTab(byManualTab, "Manual"); byManualTab->setObjectName("Manual");
// // 	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
// // 	connect(buttonBox, SIGNAL(accepted()), this, SLOT(Click_ok()));
// // 	connect(buttonBox, SIGNAL(rejected()), this, SLOT(Click_cancel()));
// // 	connect(LERadius, SIGNAL(editingFinished()), this, SLOT(particleInformation()));
// // 	QVBoxLayout *mainLayout = new QVBoxLayout;
// // 	mainLayout->addWidget(tabWidget);
// // 	mainLayout->addWidget(buttonBox);
// // 	this->setLayout(mainLayout);
// // 
// // 	this->exec();
// // 	particle_system* ps = NULL;
// // 	if (isDialogOk){
// // 		ps = _md->makeParticleSystem(LEName->text());
// // 		object* b_obj = _md->objectFromStr(CBGeometry->currentText());
// // 		if (ps->isMemoryAlloc()){
// // 			ps->addParticles(b_obj);
// // 		}
// // 		else{
// // 			ps->makeParticles(b_obj, LESpacing->text().toDouble(), LERadius->text().toDouble());
// // 			ps->setCollision(LERestitution->text().toDouble(), LEFriction->text().toDouble(), LERollingFriction->text().toDouble(), LECohesion->text().toDouble(), 0.8f);
// // 		}
// // 		
// // 		//ps->makeParticles
// // 	}
// // 	return ps;
// // }
// // 
// // void particleDialog::Click_ok()
// // {
// // 	bool isManual;
// // 	float p[4] = { 0, };
// // 	float v[4] = { 0, };
// // 	QWidget *cTab = tabWidget->currentWidget();
// // 	QString on = cTab->objectName();
// // 	if (on == "Geometry"){
// // 		if (LEName->text().isEmpty()){
// // 			msgBox("Value of name is empty!!", QMessageBox::Critical);
// // 			return;
// // 		}
// // 		if (LERadius->text().isEmpty()){
// // 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// // 			return;
// // 		}
// // 		if (LERestitution->text().isEmpty()){
// // 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// // 			return;
// // 		}
// // // 		if (LEShearModulus->text().isEmpty()){
// // // 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// // // 			return;
// // // 		}
// // 		if (LEFriction->text().isEmpty()){
// // 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// // 			return;
// // 		}
// // 	//	baseGeometry = CBGeometry->currentText();
// // 		//this->setObjectName(LEName->text());// = LEName->text();
// // 		isManual = false;
// // 		//radius = LERadius->text().toDouble();
// // 		// 		QString comm;
// // 		// 		QTextStream(&comm) << "Create Geometry " << baseGeometry << " " << radius << "\n";
// // 	}
// // 	else if (on == "Manual"){
// // // 		QStringList ss;// = LEPa->text().split(",");
// // // 		if (LEMRadius->text().isEmpty()){
// // // 			msgBox("Value of radius is empty!!", QMessageBox::Critical);
// // // 			return;
// // // 		}
// // // 		if (LEPosition->text().split(",").size() == 3)
// // // 			ss = LEPosition->text().split(",");
// // // 		else
// // // 			if (LEPosition->text().split(" ").size() == 3)
// // // 				ss = LEPosition->text().split(" ");
// // // 			else
// // // 				if (LEPosition->text().split(", ").size() == 3)
// // // 					ss = LEPosition->text().split(", ");
// // // 				else {
// // // 					msgBox("Position is wrong data.", QMessageBox::Critical);
// // // 					return;
// // // 				}
// // // 				p[0] = ss.at(0).toDouble(); p[1] = ss.at(1).toDouble(); p[2] = ss.at(2).toDouble();
// // // 
// // // 				if (LEVelocity->text().split(",").size() == 3)
// // // 					ss = LEVelocity->text().split(",");
// // // 				else
// // // 					if (LEVelocity->text().split(" ").size() == 3)
// // // 						ss = LEVelocity->text().split(" ");
// // // 					else
// // // 						if (LEVelocity->text().split(", ").size() == 3)
// // // 							ss = LEVelocity->text().split(", ");
// // // 						else{
// // // 							msgBox("Velocity is wrong data.", QMessageBox::Critical);
// // // 							return;
// // // 						}
// // // 						v[0] = ss.at(0).toDouble(); v[1] = ss.at(1).toDouble(); v[2] = ss.at(2).toDouble();
// // // 						isManual = true;
// // // // 						baseGeometry = "none";
// // // // 						Object::name = "particles";
// // // // 						radius = p[3] = LEMRadius->text().toDouble();
// // // // 						mtype = material_str2enum(CBMaterial->currentText().toStdString());
// // // // 						material = getMaterialConstant(mtype);
// // // // 						AddParticleFromManual(p, v);
// // // 						// 		QString comm;
// // // 						// 		QTextStream(&comm) 
// // // 						// 			<< "Create Manual " << radius << " " << (int)mtype << " "
// // // 						// 			<< p[0] << " " << p[1] << " " << p[2] << " "
// // // 						// 			<< v[0] << " " << p[1] << " " << p[2] << "\n";
// // // 						// 		cpProcess.push_back(comm);
// // 	}
// // 
// // 	this->close();
// // 	isDialogOk = true;
// // }
// // 
// // void particleDialog::Click_cancel()
// // {
// // 	this->close();
// // 	isDialogOk = false;
// // }
// // 
// // void particleDialog::particleInformation()
// // {
// // 	object* obj = md->objectFromStr(CBGeometry->currentText());
// // 	float rad = LERadius->text().toDouble();
// // 	unsigned int np = obj->makeParticles(rad, LESpacing->text().toDouble(), true);
// // 	float ms = obj->density() * 4.0f * (float)M_PI * pow(rad, 3.0f) / 3.0f;
// // 	float tms = ms * np;
// // 	QString tx;
// // 	QTextStream(&tx) << np;
// // 	LENumParticle->setText(tx); tx.clear();
// // 	QTextStream(&tx) << tms;
// // 	LETotalMass->setText(tx);
// // }