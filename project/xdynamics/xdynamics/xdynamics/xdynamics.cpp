#include "xdynamics.h"
#include "vparticles.h"
#include "vcontroller.h"
#include "solveDialog.h"
#include "contactCoefficientTable.h"
#include "ExportDialog.h"
#include "cubeDialog.h"
#include "cylinderDialog.h"
#include "newDialog.h"
//#include "polygonDialog.h"
#include "planeDialog.h"
#include "massDialog.h"
#include "particleDialog.h"
//#include "ccDialog.h"
#include "hmcmDialog.h"
#include "objProperty.h"
#include "dembd_simulation.h"
#include "msgBox.h"
#include "database.h"
//#include "solveProcess.h"
#include <QThread>
#include <QDebug>
#include <QtWidgets>
#include <QFileDialog.h>

//using namespace xdynamics;

xdynamics::xdynamics(int argc, char** argv, QWidget *parent)
	: QMainWindow(parent)
	, _isOnMainActions(false)
	, md(NULL)
	, th(NULL)
	, sim(NULL)
	, pBar(NULL)
	, gl(NULL)
	, ptObj(NULL)
	, db(NULL)
	, cmd(NULL)
{
	animation_statement = false;
	ui.setupUi(this);
	gl = new GLWidget(argc, argv, NULL);
	ui.GraphicArea->setWidget(gl);
	QMainWindow::show();
	
	setBaseAction();
	newproj();
}

xdynamics::~xdynamics()
{
	if (th){
		exitThread();
	}
	if (gl) delete gl; gl = NULL;
	if (md) delete md; md = NULL;
	if (sim) delete sim; sim = NULL;
	if (th) delete th; th = NULL;
	if (ptObj) delete ptObj; ptObj = NULL;
	if (db) delete db; db = NULL;
	if (cmd) delete cmd; cmd = NULL;
}

void xdynamics::setBaseAction()
{
	newAct = new QAction(QIcon(":/Resources/new.png"), tr("&New"), this);
	newAct->setStatusTip(tr("New"));
	connect(newAct, SIGNAL(triggered()), this, SLOT(newproj()));

	openAct = new QAction(QIcon(":/Resources/open.png"), tr("&Open"), this);
	openAct->setStatusTip(tr("Open project"));
	connect(openAct, SIGNAL(triggered()), this, SLOT(openproj()));

	saveAct = new QAction(QIcon(":/Resources/save.png"), tr("&Save"), this);
	saveAct->setStatusTip(tr("Save project"));
	connect(saveAct, SIGNAL(triggered()), this, SLOT(saveproj()));

	ui.mainToolBar->addAction(newAct);
	ui.mainToolBar->addAction(openAct);
	ui.mainToolBar->addAction(saveAct);

	connect(ui.actionChange_Shape, SIGNAL(triggered()), this, SLOT(ChangeShape()));
	connect(ui.actionDEM_Result_ASCII, SIGNAL(triggered()), this, SLOT(DEMRESULTASCII_Export()));
	connect(ui.actionMilkShape_3D_ASCII, SIGNAL(triggered()), this, SLOT(MS3DASCII_Import()));
	connect(ui.actionMBD_Result_ASCII, SIGNAL(triggered()), this, SLOT(MBDRESULTASCII_Export()));
	connect(ui.actionProperty, SIGNAL(triggered()), this, SLOT(OBJPROPERTY_Dialog()));
}

void xdynamics::setMainAction()
{
	makeCubeAct = new QAction(QIcon(":/Resources/pRec.png"), tr("&Create Cube Object"), this);
	makeCubeAct->setStatusTip(tr("Create Cube Object"));
	connect(makeCubeAct, SIGNAL(triggered()), this, SLOT(makeCube()));

	makeRectAct = new QAction(QIcon(":/Resources/icRect.png"), tr("&Create Rectangle Object"), this);
	makeRectAct->setStatusTip(tr("Create Rectangle Object"));
	connect(makeRectAct, SIGNAL(triggered()), this, SLOT(makePlane()));

	makeLineAct = new QAction(QIcon(":/Resources/icLine.png"), tr("&Create Line Object"), this);
	makeLineAct->setStatusTip(tr("Create Line Object"));
	connect(makeLineAct, SIGNAL(triggered()), this, SLOT(makeLine()));

	makePolyAct = new QAction(QIcon(":/Resources/icPolygon.png"), tr("&Create Polygon Object"), this);
	makePolyAct->setStatusTip(tr("Create Polygon Object"));
	connect(makePolyAct, SIGNAL(triggered()), this, SLOT(makePolygon()));

	makeCylinderAct = new QAction(QIcon(":/Resources/cylinder.png"), tr("&Create Cylinder Object"), this);
	makeCylinderAct->setStatusTip(tr("Create Cylinder Object"));
	connect(makeCylinderAct, SIGNAL(triggered()), this, SLOT(makeCylinder()));

	makeParticleAct = new QAction(QIcon(":/Resources/particle.png"), tr("&Create particles"), this);
	makeParticleAct->setStatusTip(tr("Create particles"));
	connect(makeParticleAct, SIGNAL(triggered()), this, SLOT(makeParticle()));

	makeMassAct = new QAction(QIcon(":/Resources/mass.png"), tr("&Create mass"), this);
	makeMassAct->setStatusTip(tr("Create mass"));
	connect(makeMassAct, SIGNAL(triggered()), this, SLOT(makeMass()));

	collidConstAct = new QAction(QIcon(":/Resources/collision.png"), tr("&Create Contact Element"), this);
	collidConstAct->setStatusTip(tr("Create Contact Element"));
	connect(collidConstAct, SIGNAL(triggered()), this, SLOT(makeHMCM()));

	solveProcessAct = new QAction(QIcon(":/Resources/solve.png"), tr("&Solve the model"), this);
	solveProcessAct->setStatusTip(tr("Solve the model"));
	connect(solveProcessAct, SIGNAL(triggered()), this, SLOT(solve()));

	changeParticleAct = new QAction(QIcon(":/Resources/icChangeParticle.png"), tr("&Change particle from file"), this);
	changeParticleAct->setStatusTip(tr("Change particles from file"));
	connect(changeParticleAct, SIGNAL(triggered()), this, SLOT(ChangeParticleFromFile()));
	
	projectionViewAct = new QAction(QIcon(":/Resources/perspective.png"), tr("&Change perspective view mode"), this);
	projectionViewAct->setStatusTip(tr("Change perspective view mode"));
	connect(projectionViewAct, SIGNAL(triggered()), this, SLOT(changeProjectionViewMode()));

	paletteAct = new QAction(QIcon(":/Resources/sketch.png"), tr("&Change sketch mode"), this);
	paletteAct->setStatusTip(tr("Change sketch mode"));
	connect(paletteAct, SIGNAL(triggered()), this, SLOT(changePaletteMode()));

	ui.mainToolBar->addAction(makeCubeAct);
	ui.mainToolBar->addAction(makeRectAct);
	ui.mainToolBar->addAction(makeLineAct);
	ui.mainToolBar->addAction(makePolyAct);
	ui.mainToolBar->addAction(makeCylinderAct);
	ui.mainToolBar->addAction(makeParticleAct);
	ui.mainToolBar->addAction(makeMassAct);
	ui.mainToolBar->addAction(changeParticleAct);
	ui.mainToolBar->addAction(collidConstAct);
	ui.mainToolBar->addAction(solveProcessAct);
	ui.mainToolBar->insertSeparator(solveProcessAct);
	ui.mainToolBar->addAction(projectionViewAct);
	ui.mainToolBar->addAction(paletteAct);

	viewObjectComboBox = new QComboBox;
	viewObjectComboBox->insertItem(0, "All display");
	viewObjectComboBox->insertItem(1, "Only frame");
	viewObjectComboBox->insertItem(2, "Only particle");
	connect(viewObjectComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(ChangeComboBox(int)));
	ui.secToolBar->addWidget(viewObjectComboBox);

	aniPreviousAct2 = new QAction(QIcon(":/Resources/ani_moreprevious.png"), tr("&previous2x"), this);
	aniPreviousAct2->setStatusTip(tr("2x previous for animation."));
	connect(aniPreviousAct2, SIGNAL(triggered()), this, SLOT(ani_previous2x()));

	aniPreviousAct = new QAction(QIcon(":/Resources/ani_previous.png"), tr("&previous1x"), this);
	aniPreviousAct->setStatusTip(tr("1x previous for animation."));
	connect(aniPreviousAct, SIGNAL(triggered()), this, SLOT(ani_previous1x()));

	aniPlayAct = new QAction(QIcon(":/Resources/ani_play.png"), tr("&play"), this);
	aniPlayAct->setStatusTip(tr("play for animation."));
	connect(aniPlayAct, SIGNAL(triggered()), this, SLOT(ani_play()));

	simPlayAct = new QAction(QIcon(":/Resources/ani_play.png"), tr("&simulation"), this);
	simPlayAct->setStatusTip(tr("control simulation"));
	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_play()));

	aniForwardAct = new QAction(QIcon(":/Resources/ani_fast.png"), tr("&forward1x"), this);
	aniForwardAct->setStatusTip(tr("1x forward for animation."));
	connect(aniForwardAct, SIGNAL(triggered()), this, SLOT(ani_forward1x()));

	aniForwardAct2 = new QAction(QIcon(":/Resources/ani_morefast.png"), tr("&forward2x"), this);
	aniForwardAct2->setStatusTip(tr("2x forward for animation."));
	connect(aniForwardAct2, SIGNAL(triggered()), this, SLOT(ani_forward2x()));

	ui.secToolBar->addAction(aniPreviousAct2);
	ui.secToolBar->addAction(aniPreviousAct);
	ui.secToolBar->addAction(aniPlayAct);
	ui.secToolBar->addAction(simPlayAct);
	simPlayAct->setEnabled(false);
	ui.secToolBar->addAction(aniForwardAct);
	ui.secToolBar->addAction(aniForwardAct2);
	setAnimationAction(false);
	HSlider = new QSlider(Qt::Orientation::Horizontal, this);

	HSlider->setFixedWidth(100);
	connect(HSlider, SIGNAL(valueChanged(int)), this, SLOT(ani_scrollbar()));
	ui.secToolBar->addWidget(HSlider);

	LEframe = new QLineEdit(this);
	LEframe->setText(QString("0"));
	LEframe->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LEframe->setContentsMargins(QMargins(5, 0, 0, 0));
	ui.secToolBar->addWidget(LEframe);
	Lframe = new QLabel(this);
	Lframe->setText(QString("/ 0"));
	Lframe->setContentsMargins(QMargins(5, 0, 0, 0));
	ui.secToolBar->addWidget(Lframe);
	LETimes = new QLineEdit(this);
	LETimes->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LETimes->setFixedWidth(50);
	QLabel *LTimes = new QLabel(this);
	LTimes->setText(QString("Time : "));
	LTimes->setContentsMargins(QMargins(10, 0, 0, 0));
	ui.secToolBar->addWidget(LTimes);
	ui.secToolBar->addWidget(LETimes);
}

void xdynamics::ChangeComboBox(int id)
{
	gl->ChangeDisplayOption(id);
}

void xdynamics::newproj()
{
	newDialog *nd = new newDialog;
	if (nd->callDialog())
	{
		setMainAction();
		_isOnMainActions = true;
		QString nm = nd->name();
		QString pt = nd->path();
		tUnit u = nd->unit();
		tGravity dg = nd->gravityDirection();
		md = new modeler(pt + nm, DEM, u, dg);
		db = new database(this, md);
		addDockWidget(Qt::RightDockWidgetArea, db);
	}
	else
	{
		if (!md)
			md = new modeler;
		db = new database(this, md);
		addDockWidget(Qt::RightDockWidgetArea, db);
		md->openModeler(gl, nd->fullPath());
		//ui.PTEdit->appendPlainText("Open file : " + nd->fullPath());
		if (!_isOnMainActions)
			setMainAction();
		_isOnMainActions = true;
	}
	cmd = new cmdWindow(this);
	addDockWidget(Qt::BottomDockWidgetArea, cmd);
	gl->setModeler(md);
	//md->bindingCommand(ui.PTEdit);
	delete nd;
}

// CODEDYN
void xdynamics::openproj()
{
	QString dir;
	if (md)
		dir = md->modelPath();
	else
		dir = "C:/mphysics";
	QStringList fileNames = QFileDialog::getOpenFileNames(this, tr("open"), dir, tr("DEM result file (*.bin);;MBD result file (*.mrf);;DEM model file (*.mde);;SPH model file (*.sph)"));

	if (fileNames.isEmpty())
		return;
	if (fileNames.size() == 1){
		QString file = fileNames.at(0);
		//md->comm("Open file : " + file);
		int begin = file.lastIndexOf(".");
		QString ext = file.mid(begin);
		if (ext == ".mde"){
			if (!md)
				md = new modeler;
			md->openModeler(gl, file);
			if (!_isOnMainActions)
				setMainAction();
		}
		if (ext == ".sph"){
			gl->openSph(file);
		}
		if (ext == ".mrf"){
			gl->openMbd(file);
			setAnimationAction(true);
		}
	}
	else{
		QString file = fileNames.at(0);
// 		QString str;
// 		QTextStream(&str) << "Open files : " + file + " ~ " + file.mid(file.lastIndexOf("/"));
		//md->comm("Open files : " + file + " ~ " + fileNames.at(fileNames.size() - 1).mid(file.lastIndexOf("/")));
		int begin = file.lastIndexOf(".");
		QString ext = file.mid(begin);
		if (ext == ".bin")
			gl->openResults(fileNames);
		setAnimationAction(true);
	}
}

void xdynamics::setAnimationAction(bool b)
{
	aniPreviousAct2->setEnabled(b);
	aniPreviousAct->setEnabled(b);
	aniPlayAct->setEnabled(b);
	//simPlayAct->setDisabled(true);
	aniForwardAct->setEnabled(b);
	aniForwardAct2->setEnabled(b);
}

void xdynamics::saveproj()
{
	md->saveModeler();
}

void xdynamics::mySlot()
{
	int cf = vcontroller::getFrame();

	HSlider->setValue(cf);
	QString str;
	str.sprintf("%d", cf);
	LEframe->setText(str);
	float time = vcontroller::getTimes();
	str.clear(); str.sprintf("%f", time);
	LETimes->setText(str);
}

void xdynamics::ani_previous2x()
{
	if (gl->is_set_particle()){
		ani_pause();
		vcontroller::move2previous2x();
		QString tf;
		tf.sprintf("%.5f", vcontroller::getTimes());
		LETimes->setText(tf);
		mySlot();
	}
}

void xdynamics::ani_previous1x()
{
	if (gl->is_set_particle()){
		ani_pause();
		vcontroller::move2previous1x();
		QString tf;
		tf.sprintf("%.5f", vcontroller::getTimes());
		LETimes->setText(tf);
		mySlot();
	}
}

void xdynamics::ani_play()
{
	animation_statement = true;
	if (gl->is_set_particle()){
		vcontroller::on_play();
		QString tf;
		tf.sprintf("/ %d", vcontroller::getTotalBuffers() - 1);
		Lframe->setText(tf);
		HSlider->setMaximum(vcontroller::getTotalBuffers() - 1);
		connect(gl, SIGNAL(mySignal()), this, SLOT(mySlot()));
	}

	if (vcontroller::is_end_frame())
		return;

	disconnect(aniPlayAct);
	aniPlayAct->setIcon(QIcon(":/Resources/ani_pause.png"));
	aniPlayAct->setStatusTip(tr("Pause for animation."));
	connect(aniPlayAct, SIGNAL(triggered()), this, SLOT(ani_pause()));
}

void xdynamics::sim_play()
{
	disconnect(simPlayAct);
	simPlayAct->setIcon(QIcon(":/Resources/ani_pause.png"));
	simPlayAct->setStatusTip(tr("Restart for simulation."));
	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_stop()));
	//simPlayAct->setEnabled(true);
	sim->setWaitSimulation(false);
	setAnimationAction(false);
}

void xdynamics::sim_stop()
{
	disconnect(simPlayAct);
	simPlayAct->setIcon(QIcon(":/Resources/ani_play.png"));
	simPlayAct->setStatusTip(tr("Restart for simulation."));
	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_play()));
	//simPlayAct->setEnabled(true);
	sim->setWaitSimulation(true);
	setAnimationAction(true);
}

void xdynamics::ani_pause()
{
	animation_statement = false;
	vcontroller::off_play();
	disconnect(aniPlayAct);
	aniPlayAct->setIcon(QIcon(":/Resources/ani_play.png"));
	aniPlayAct->setStatusTip(tr("Play for animation."));
	connect(aniPlayAct, SIGNAL(triggered()), this, SLOT(ani_play()));
	disconnect(gl, SIGNAL(mySignal()), this, SLOT(mySlot()));
}

void xdynamics::ani_forward1x()
{
	if (gl->is_set_particle()) {
		ani_pause();
		vcontroller::move2forward1x();
		QString tf;
		//	double t = view_controller::getTimes();
		tf.sprintf("%.5f", vcontroller::getTimes());
		LETimes->setText(tf);
		mySlot();
	}
}

void xdynamics::ani_forward2x()
{
	if (gl->is_set_particle()) {
		ani_pause();
		vcontroller::move2forward2x();
		QString tf;
		tf.sprintf("%.5f", vcontroller::getTimes());
		LETimes->setText(tf);
		mySlot();
	}
}

void xdynamics::ani_scrollbar()
{
	if (animation_statement){
		QString tf;
		tf.sprintf("%.5f", vcontroller::getTimes());
		LETimes->setText(tf);
		return;
	}

	int value = HSlider->value();
	QString str;
	str.sprintf("%d", value);
	LEframe->setText(str);
	if (gl->is_set_particle()) {
		vcontroller::setFrame(unsigned int(value));
		// 		if (view_controller::getRealTimeParameter())
		// 			gl->UpdateRtDEMData();
	}
}

void xdynamics::openPinfoDialog()
{
	// 	if (animation_statement)
	// 		return;
	// 	if (gl->is_set_particle()){
	// 		if (!pinfoDialog)
	// 			pinfoDialog = new particleInfoDialog(this);
	// 
	// 		pinfoDialog->bindingParticleViewer(gl->vcontroller());
	// 		pinfoDialog->show();
	// 		pinfoDialog->raise();
	// 		pinfoDialog->activateWindow();
	// 	}

}

void xdynamics::ChangeParticleFromFile()
{
	QString dir = md->modelPath();
	QString file = QFileDialog::getOpenFileName(this, tr("open"), dir);
	if (file.isEmpty())
		return;
	if (gl->change(file, CHANGE_PARTICLE_POSITION, BIN)){
		md->particleSystem()->changeParticlesFromVP(gl->vParticles()->getPosition());
	}
}

void xdynamics::ChangeShape()
{

}

void xdynamics::DEMRESULTASCII_Export()
{
	ExportDialog edlg(md);
	edlg.callDialog();
}

void xdynamics::MBDRESULTASCII_Export()
{
	QString dir = md->modelPath();
	QString fileName = QFileDialog::getOpenFileName(this, tr("Export"), dir, tr("MBD Result File(*.mrf)"));
	QFile qfi(fileName);
	QString txtFile = md->modelPath() + "/" + md->modelName() + "_mrf.txt";
	qfi.open(QIODevice::ReadOnly);
	QFile qfo(txtFile);
	qfo.open(QIODevice::WriteOnly);
	QTextStream qso(&qfo);

 	unsigned int cnt = 0;
	unsigned int nm = 0;
	unsigned int id = 0;
	unsigned int nout = 0;
	unsigned int name_size = 0;
	double ct = 0.0;
	char ch;
	
	VEC3D p, v, a;
	EPD ep, ev, ea;
	qfi.read((char*)&nm, sizeof(unsigned int));
	qfi.read((char*)&nout, sizeof(unsigned int));
	while (!qfi.atEnd()){
		qfi.read((char*)&cnt, sizeof(unsigned int));
		for (unsigned int i = 0; i < nm; i++){
			QString str;
			qfi.read((char*)&name_size, sizeof(unsigned int));
			for (unsigned int j = 0; j < name_size; j++){
				qfi.read(&ch, sizeof(char));
				str.push_back(ch);
			}
			qfi.read((char*)&id, sizeof(unsigned int));
			qfi.read((char*)&ct, sizeof(double));
			qfi.read((char*)&p, sizeof(VEC3D));
			qfi.read((char*)&ep, sizeof(EPD));
			qfi.read((char*)&v, sizeof(VEC3D));
			qfi.read((char*)&ev, sizeof(EPD));
			qfi.read((char*)&a, sizeof(VEC3D));
			qfi.read((char*)&ea, sizeof(EPD));
			qso << ct << " " << str << " " << p.x << " " << p.y << " " << p.z
				<< " " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3
				<< " " << v.x << " " << v.y << " " << v.z
				<< " " << ev.e0 << " " << ev.e1 << " " << ev.e2 << " " << ev.e3
				<< " " << a.x << " " << a.y << " " << a.z
				<< " " << ea.e0 << " " << ea.e1 << " " << ea.e2 << " " << ea.e3 << " ";
		}
		qso << endl;
	}

	qfo.close();
	qfi.close();
}

void xdynamics::MS3DASCII_Import()
{
	QString dir = md->modelPath();
	QString fileName = QFileDialog::getOpenFileName(this, tr("Import"), dir);
	polygonObject* po = md->makePolygonObject(MILKSHAPE_3D_ASCII, fileName);
	gl->makePolygonObject(po);

	//gl->addParticles
}

void xdynamics::OBJPROPERTY_Dialog()
{
	if (!ptObj)
		ptObj = new objProperty(this);
	ptObj->initialize(md, gl);
	ptObj->show();
}

void xdynamics::makeCube()
{
	cubeDialog cd;
	gl->makeCube(cd.callDialog(md));
}

void xdynamics::makePlane()
{
	planeDialog pd;
	gl->makePlane(pd.callDialog(md));
	
}

void xdynamics::makePolygon()
{
	//	polygonDialog pd;
	//	gl->makePolygon(pd.callDialog(md));
}

void xdynamics::makeCylinder()
{
	cylinderDialog cyd;
	gl->makeCylinder(cyd.callDialog(md));
}

void xdynamics::makeLine()
{
	gl->makeLine();
}

void xdynamics::makeParticle()
{
// 	particleDialog pd;
// 	gl->makeParticle(pd.callDialog(md));
	particleDialog pd(this, md);
	gl->makeParticle(md->particleSystem());
}

void xdynamics::makeMass()
{
	massDialog msd;
	mass* ms = msd.callDialog(md);
	if (ms)
	{
		if (ms->getBaseGeometryType() == POLYGON)
			ms->setPolyGeometryObject(gl->getVPolyObjectFromName(ms->name()));
		else
			ms->setGeometryObject(gl->getVObjectFromName(ms->name()));
	}
		//ms->setgl->getVObjectFromName(ms->name());
		//gl->makeMassCoordinate(ms->name());
}

void xdynamics::makeHMCM()
{
	hmcmDialog hmcm(this, md);
}

void xdynamics::exitThread()
{
	if (th->isRunning()){
		ui.statusBar->removeWidget(pBar);
		delete pBar;
		pBar = NULL;

		th->exit();
		delete sim;
		sim = NULL;
		simPlayAct->setDisabled(true);
	}
}

void xdynamics::recieveProgress(unsigned int pt)
{
	pBar->setValue(pt);
	//pBar->update();
}

void xdynamics::deleteFileByEXT(QString ext)
{
	QString dDir = md->modelPath() + "/";
	QDir dir = QDir(dDir);
	QStringList delFileList;
	delFileList = dir.entryList(QStringList("*." + ext), QDir::Files | QDir::NoSymLinks);
	qDebug() << "The number of *.bin file : " << delFileList.length();
	for (int i = 0; i < delFileList.length(); i++){
		QString deleteFilePath = dDir + delFileList[i];
		QFile::remove(deleteFilePath);
	}
	qDebug() << "Complete delete.";
}

void xdynamics::waitSimulation()
{
	disconnect(simPlayAct);
	simPlayAct->setIcon(QIcon(":/Resources/ani_play.png"));
	simPlayAct->setStatusTip(tr("Restart for simulation."));
	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_play()));
	//simPlayAct->setEnabled(true);
	sim->wait();
}

void xdynamics::solve()
{
	solveDialog sd;
	if (!sd.callDialog())
		return;

	qDebug() << "Delete *.bin files of " << md->modelPath();
	deleteFileByEXT("bin");

	if (th == NULL)
		th = new QThread;
	if (sim == NULL){
		if (md->numParticle() && !md->numMass()){
			sim = new dem_simulation(md);
		}
		else if (!md->numParticle() && md->numMass()){
			sim = new mbd_simulation(md);
		}
		else if (md->numParticle() && md->numMass()){
			sim = new dembd_simulation(md, new dem_simulation(md), new mbd_simulation(md));
		}
	}

	sim->setSimulationCondition(sd.simTime, sd.timeStep, sd.saveStep);
	if (sim->initialize(sd.isCpu)){
		sim->setCommandWindow(cmd);
		cmd->write(CMD_INFO, QString("Simulation is initialized."));
		//qDebug() << "- Initialization of simulation ---------------------------- DONE";
		if (pBar == NULL)
			pBar = new QProgressBar;
		pBar->setMaximum(sim->numStep() / sd.saveStep);
		ui.statusBar->addWidget(pBar, 1);
		sim->moveToThread(th);
		connect(sim, SIGNAL(finished()), this, SLOT(exitThread()));
		connect(sim, SIGNAL(sendProgress(unsigned int)), this, SLOT(recieveProgress(unsigned int)));
		//connect(newAct, SIGNAL(triggered()), this, SLOT(waitSimulation()));
		if (sd.isCpu){
			connect(th, SIGNAL(started()), sim, SLOT(cpuRun()));
		}
		else{
			connect(th, SIGNAL(started()), sim, SLOT(gpuRun()));
		}
	}
	else{
		delete sim;
		return;
	}
	disconnect(simPlayAct);
	simPlayAct->setIcon(QIcon(":/Resources/ani_pause.png"));
	simPlayAct->setStatusTip(tr("Pause for simulation."));
	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_stop()));
	simPlayAct->setEnabled(true);
	th->start();
	qDebug() << "- Simulation Thread On (CPU) -";
	// 	if (neigh) delete neigh; neigh = NULL;
	// 	if (vv) delete vv; vv = NULL;
}

void xdynamics::changePaletteMode()
{
	bool iss = gl->changePaletteMode();
	if (iss)
		paletteAct->setIcon(QIcon(":/Resources/noSketch.png"));
	else
		paletteAct->setIcon(QIcon(":/Resources/sketch.png"));
}

void xdynamics::changeProjectionViewMode()
{
	projectionType pt = gl->changeProjectionViewMode();
	if (pt == ORTHO_PROJECTION)
		projectionViewAct->setIcon(QIcon(":/Resources/perspective.png"));
	else if (pt == PERSPECTIVE_PROJECTION)
		projectionViewAct->setIcon(QIcon(":/Resources/ortho.png"));
}