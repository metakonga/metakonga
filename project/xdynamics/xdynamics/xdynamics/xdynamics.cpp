#include "xdynamics.h"
#include "dialogs.h"
#include "Objects.h"
#include "vparticles.h"
#include "vcontroller.h"
#include "solveDialog.h"
#include "contactCoefficientTable.h"
#include "ExportDialog.h"
#include "cylinderDialog.h"
#include "newDialog.h"
//#include "polygonDialog.h"
#include "massDialog.h"
//#include "ccDialog.h"
#include "hmcmDialog.h"
#include "objProperty.h"
#include "xDynamicsSolver.h"
#include "msgBox.h"
#include "database.h"
//#include "solveProcess.h"
#include <QThread>
#include <QDebug>
#include <QtWidgets>
#include <QFileDialog.h>

//using namespace xdynamics;
static xDynamicsSolver *solver = NULL;

xdynamics::xdynamics(int argc, char** argv, QWidget *parent)
	: QMainWindow(parent)
	, _isOnMainActions(false)
	, operating_animation(-1)
	, mg(NULL)
	, th(NULL)
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
	createMainOperations();
	model::path += PROGRAM_NAME;
	newproj();
}

xdynamics::~xdynamics()
{
	if (gl) delete gl; gl = NULL;
	if (mg) delete mg; mg = NULL;	
	if (th) delete th; th = NULL;
	if (ptObj) delete ptObj; ptObj = NULL;
	if (db) delete db; db = NULL;
	if (cmd) delete cmd; cmd = NULL;
	if (solver) delete solver; solver = NULL;
}

void xdynamics::setBaseAction()
{
// 	newAct = new QAction(QIcon(":/Resources/new.png"), tr("&New"), this);
// 	newAct->setStatusTip(tr("New"));
// 	connect(newAct, SIGNAL(triggered()), this, SLOT(newproj()));
// 
// 	openAct = new QAction(QIcon(":/Resources/open.png"), tr("&Open"), this);
// 	openAct->setStatusTip(tr("Open project"));
// 	connect(openAct, SIGNAL(triggered()), this, SLOT(openproj()));
// 
// 	saveAct = new QAction(QIcon(":/Resources/save.png"), tr("&Save"), this);
// 	saveAct->setStatusTip(tr("Save project"));
// 	connect(saveAct, SIGNAL(triggered()), this, SLOT(saveproj()));
// 
// 	ui.mainToolBar->addAction(newAct);
// 	ui.mainToolBar->addAction(openAct);
// 	ui.mainToolBar->addAction(saveAct);

	connect(ui.actionChange_Shape, SIGNAL(triggered()), this, SLOT(ChangeShape()));
	connect(ui.actionDEM_Result_ASCII, SIGNAL(triggered()), this, SLOT(DEMRESULTASCII_Export()));
	connect(ui.actionMBD_Result_ASCII, SIGNAL(triggered()), this, SLOT(MBDRESULTASCII_Export()));
	connect(ui.actionProperty, SIGNAL(triggered()), this, SLOT(OBJPROPERTY_Dialog()));
}

void xdynamics::setMainAction()
{
// 	makeCubeAct = new QAction(QIcon(":/Resources/pRec.png"), tr("&Create Cube Object"), this);
// 	makeCubeAct->setStatusTip(tr("Create Cube Object"));
// 	connect(makeCubeAct, SIGNAL(triggered()), this, SLOT(makeCube()));
// 
// 	makeRectAct = new QAction(QIcon(":/Resources/icRect.png"), tr("&Create Rectangle Object"), this);
// 	makeRectAct->setStatusTip(tr("Create Rectangle Object"));
// 	connect(makeRectAct, SIGNAL(triggered()), this, SLOT(makePlane()));
// 
// 	makeLineAct = new QAction(QIcon(":/Resources/icLine.png"), tr("&Create Line Object"), this);
// 	makeLineAct->setStatusTip(tr("Create Line Object"));
// 	connect(makeLineAct, SIGNAL(triggered()), this, SLOT(makeLine()));
// 
// 	makePolyAct = new QAction(QIcon(":/Resources/icPolygon.png"), tr("&Create Polygon Object"), this);
// 	makePolyAct->setStatusTip(tr("Create Polygon Object"));
// 	connect(makePolyAct, SIGNAL(triggered()), this, SLOT(makePolygon()));
// 
// 	makeCylinderAct = new QAction(QIcon(":/Resources/cylinder.png"), tr("&Create Cylinder Object"), this);
// 	makeCylinderAct->setStatusTip(tr("Create Cylinder Object"));
// 	connect(makeCylinderAct, SIGNAL(triggered()), this, SLOT(makeCylinder()));
// 
// 	makeParticleAct = new QAction(QIcon(":/Resources/particle.png"), tr("&Create particles"), this);
// 	makeParticleAct->setStatusTip(tr("Create particles"));
// 	connect(makeParticleAct, SIGNAL(triggered()), this, SLOT(makeParticle()));
// 
// 	makeMassAct = new QAction(QIcon(":/Resources/mass.png"), tr("&Create mass"), this);
// 	makeMassAct->setStatusTip(tr("Create mass"));
// 	connect(makeMassAct, SIGNAL(triggered()), this, SLOT(makeMass()));
// 
// 	collidConstAct = new QAction(QIcon(":/Resources/collision.png"), tr("&Create Contact Element"), this);
// 	collidConstAct->setStatusTip(tr("Create Contact Element"));
// 	connect(collidConstAct, SIGNAL(triggered()), this, SLOT(makeContactPair()));
// 
// 	solveProcessAct = new QAction(QIcon(":/Resources/solve.png"), tr("&Solve the model"), this);
// 	solveProcessAct->setStatusTip(tr("Solve the model"));
// 	connect(solveProcessAct, SIGNAL(triggered()), this, SLOT(solve()));
// 
// 	changeParticleAct = new QAction(QIcon(":/Resources/icChangeParticle.png"), tr("&Change particle from file"), this);
// 	changeParticleAct->setStatusTip(tr("Change particles from file"));
// 	connect(changeParticleAct, SIGNAL(triggered()), this, SLOT(ChangeParticleFromFile()));
// 	
// 	projectionViewAct = new QAction(QIcon(":/Resources/perspective.png"), tr("&Change perspective view mode"), this);
// 	projectionViewAct->setStatusTip(tr("Change perspective view mode"));
// 	connect(projectionViewAct, SIGNAL(triggered()), this, SLOT(changeProjectionViewMode()));
// 
// 	paletteAct = new QAction(QIcon(":/Resources/sketch.png"), tr("&Change sketch mode"), this);
// 	paletteAct->setStatusTip(tr("Change sketch mode"));
// 	connect(paletteAct, SIGNAL(triggered()), this, SLOT(changePaletteMode()));
// 
// 	ui.mainToolBar->addAction(makeCubeAct);
// 	ui.mainToolBar->addAction(makeRectAct);
// 	ui.mainToolBar->addAction(makeLineAct);
// 	ui.mainToolBar->addAction(makePolyAct);
// 	ui.mainToolBar->addAction(makeCylinderAct);
// 	ui.mainToolBar->addAction(makeParticleAct);
// 	ui.mainToolBar->addAction(makeMassAct);
// 	ui.mainToolBar->addAction(changeParticleAct);
// 	ui.mainToolBar->addAction(collidConstAct);
// 	ui.mainToolBar->addAction(solveProcessAct);
// 	ui.mainToolBar->insertSeparator(solveProcessAct);
// 	ui.mainToolBar->addAction(projectionViewAct);
// 	ui.mainToolBar->addAction(paletteAct);

// 	viewObjectComboBox = new QComboBox;
// 	viewObjectComboBox->insertItem(0, "All display");
// 	viewObjectComboBox->insertItem(1, "Only frame");
// 	viewObjectComboBox->insertItem(2, "Only particle");
// 	connect(viewObjectComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(ChangeComboBox(int)));
// 	ui.secToolBar->addWidget(viewObjectComboBox);

// 	aniPreviousAct2 = new QAction(QIcon(":/Resources/ani_moreprevious.png"), tr("&previous2x"), this);
// 	aniPreviousAct2->setStatusTip(tr("2x previous for animation."));
// 	connect(aniPreviousAct2, SIGNAL(triggered()), this, SLOT(ani_previous2x()));
// 
// 	aniPreviousAct = new QAction(QIcon(":/Resources/ani_previous.png"), tr("&previous1x"), this);
// 	aniPreviousAct->setStatusTip(tr("1x previous for animation."));
// 	connect(aniPreviousAct, SIGNAL(triggered()), this, SLOT(ani_previous1x()));
// 
// 	aniPlayAct = new QAction(QIcon(":/Resources/ani_play.png"), tr("&play"), this);
// 	aniPlayAct->setStatusTip(tr("play for animation."));
// 	connect(aniPlayAct, SIGNAL(triggered()), this, SLOT(ani_play()));
// 
// 	simPlayAct = new QAction(QIcon(":/Resources/ani_play.png"), tr("&simulation"), this);
// 	simPlayAct->setStatusTip(tr("control simulation"));
// 	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_play()));
// 
// 	aniForwardAct = new QAction(QIcon(":/Resources/ani_fast.png"), tr("&forward1x"), this);
// 	aniForwardAct->setStatusTip(tr("1x forward for animation."));
// 	connect(aniForwardAct, SIGNAL(triggered()), this, SLOT(ani_forward1x()));
// 
// 	aniForwardAct2 = new QAction(QIcon(":/Resources/ani_morefast.png"), tr("&forward2x"), this);
// 	aniForwardAct2->setStatusTip(tr("2x forward for animation."));
// 	connect(aniForwardAct2, SIGNAL(triggered()), this, SLOT(ani_forward2x()));
// 
// 	ui.secToolBar->addAction(aniPreviousAct2);
// 	ui.secToolBar->addAction(aniPreviousAct);
// 	ui.secToolBar->addAction(aniPlayAct);
// 	ui.secToolBar->addAction(simPlayAct);
// 	simPlayAct->setEnabled(false);
// 	ui.secToolBar->addAction(aniForwardAct);
// 	ui.secToolBar->addAction(aniForwardAct2);
// 	setAnimationAction(false);
// 	HSlider = new QSlider(Qt::Orientation::Horizontal, this);
// 
// 	HSlider->setFixedWidth(100);
// 	connect(HSlider, SIGNAL(valueChanged(int)), this, SLOT(ani_scrollbar()));
// 	ui.secToolBar->addWidget(HSlider);
// 
// // 	QLineEdit *sketch_space = new QLineEdit;
// // 	sketch_space->setText("0.02");
// // 	connect(sketch_space, SIGNAL(editingFinished()), gl, SLOT(setSketchSpace()));
// // 	ui.secToolBar->addWidget(sketch_space);
// 
// 	LEframe = new QLineEdit(this);
// 	LEframe->setText(QString("0"));
// 	LEframe->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
// 	LEframe->setFixedWidth(50);
// 	LEframe->setContentsMargins(QMargins(5, 0, 0, 0));
// 	ui.secToolBar->addWidget(LEframe);
// 	Lframe = new QLabel(this);
// 	Lframe->setText(QString("/ 0"));
// 	Lframe->setContentsMargins(QMargins(5, 0, 0, 0));
// 	ui.secToolBar->addWidget(Lframe);
// 	LETimes = new QLineEdit(this);
// 	LETimes->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
// 	LEframe->setFixedWidth(50);
// 	LETimes->setFixedWidth(50);
// 	QLabel *LTimes = new QLabel(this);
// 	LTimes->setText(QString("Time : "));
// 	LTimes->setContentsMargins(QMargins(10, 0, 0, 0));
// 	ui.secToolBar->addWidget(LTimes);
// 	ui.secToolBar->addWidget(LETimes);
}

void xdynamics::createMainOperations()
{
	QAction* a;

	ui.mainToolBar->setWindowTitle("Main Operations");

	a = new QAction(QIcon(":/Resources/new.png"), tr("&New"), this);
	a->setStatusTip(tr("New"));
	connect(a, SIGNAL(triggered()), this, SLOT(newproj()));
	myMainActions.insert(NEW, a);

	a = new QAction(QIcon(":/Resources/open.png"), tr("&Open"), this);
	a->setStatusTip(tr("Open project"));
	connect(a, SIGNAL(triggered()), this, SLOT(openproj()));
	myMainActions.insert(OPEN, a);

	a = new QAction(QIcon(":/Resources/save.png"), tr("&Save"), this);
	a->setStatusTip(tr("Save project"));
	connect(a, SIGNAL(triggered()), this, SLOT(saveproj()));
	myMainActions.insert(SAVE, a);

	for (int i = 0; i < myMainActions.size(); i++)
	{
		ui.mainToolBar->addAction(myMainActions.at(i));
	}
	connect(ui.Import_MilkShape_3D_ASCII, SIGNAL(triggered()), this, SLOT(MS3DASCII_Import()));
}

void xdynamics::createToolOperations()
{
	QAction *a;
	myModelingBar = addToolBar(tr("Modeling Operations"));

	a = new QAction(QIcon(":/Resources/pRec.png"), tr("&Create Cube Object"), this);
	a->setStatusTip(tr("Create Cube Object"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeCube()));
	myModelingActions.insert(MAKE_CUBE, a);

	a = new QAction(QIcon(":/Resources/icRect.png"), tr("&Create Rectangle Object"), this);
	a->setStatusTip(tr("Create Rectangle Object"));
	connect(a, SIGNAL(triggered()), this, SLOT(makePlane()));
	myModelingActions.insert(MAKE_RECT, a);

	a = new QAction(QIcon(":/Resources/icLine.png"), tr("&Create Line Object"), this);
	a->setStatusTip(tr("Create Line Object"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeLine()));
	myModelingActions.insert(MAKE_LINE, a);

	a = new QAction(QIcon(":/Resources/icPolygon.png"), tr("&Create Polygon Object"), this);
	a->setStatusTip(tr("Create Polygon Object"));
	connect(a, SIGNAL(triggered()), this, SLOT(makePolygon()));
	myModelingActions.insert(MAKE_POLY, a);

	a = new QAction(QIcon(":/Resources/cylinder.png"), tr("&Create Cylinder Object"), this);
	a->setStatusTip(tr("Create Cylinder Object"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeCylinder()));
	myModelingActions.insert(MAKE_CYLINDER, a);

	a = new QAction(QIcon(":/Resources/particle.png"), tr("&Create particles"), this);
	a->setStatusTip(tr("Create particles"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeParticle()));
	myModelingActions.insert(MAKE_PARTICLE, a);

	a = new QAction(QIcon(":/Resources/mass.png"), tr("&Create mass"), this);
	a->setStatusTip(tr("Create mass"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeMass()));
	myModelingActions.insert(MAKE_MASS, a);

	a = new QAction(QIcon(":/Resources/collision.png"), tr("&Create Contact Element"), this);
	a->setStatusTip(tr("Create Contact Element"));
	connect(a, SIGNAL(triggered()), this, SLOT(makeContactPair()));
	myModelingActions.insert(MAKE_COLLISION, a);

	a = new QAction(QIcon(":/Resources/solve.png"), tr("&Solve the model"), this);
	a->setStatusTip(tr("Solve the model"));
	connect(a, SIGNAL(triggered()), this, SLOT(solve()));
	myModelingActions.insert(RUN_ANALYSIS, a);

// 	changeParticleAct = new QAction(QIcon(":/Resources/icChangeParticle.png"), tr("&Change particle from file"), this);
// 	changeParticleAct->setStatusTip(tr("Change particles from file"));
// 	connect(changeParticleAct, SIGNAL(triggered()), this, SLOT(ChangeParticleFromFile()));

	a = new QAction(QIcon(":/Resources/perspective.png"), tr("&Change perspective view mode"), this);
	a->setStatusTip(tr("Change perspective view mode"));
	connect(a, SIGNAL(triggered()), this, SLOT(changeProjectionViewMode()));
	myModelingActions.insert(CHANGE_PROJECTION_VIEW, a);

// 	paletteAct = new QAction(QIcon(":/Resources/sketch.png"), tr("&Change sketch mode"), this);
// 	paletteAct->setStatusTip(tr("Change sketch mode"));
// 	connect(paletteAct, SIGNAL(triggered()), this, SLOT(changePaletteMode()));
	for (int i = 0; i < myModelingActions.size(); i++)
	{
		myModelingBar->addAction(myModelingActions.at(i));
	}
	myModelingBar->show();
}

void xdynamics::createAnimationOperations()
{
	myAnimationBar = addToolBar(tr("Animation Operations"));
	QAction *a;

	a = new QAction(QIcon(":/Resources/ani_tobegin.png"), tr("&toBegin"), this);
	a->setStatusTip(tr("Go to begin"));
	connect(a, SIGNAL(triggered()), this, SLOT(onGoToBegin()));
	myAnimationActions.insert(ANIMATION_GO_BEGIN, a);

	a = new QAction(QIcon(":/Resources/ani_moreprevious.png"), tr("&previous2X"), this);
	a->setStatusTip(tr("previous 2X"));
	connect(a, SIGNAL(triggered()), this, SLOT(onPrevious2X()));
	myAnimationActions.insert(ANIMATION_PREVIOUS_2X, a);

	a = new QAction(QIcon(":/Resources/ani_previous.png"), tr("&previous1X"), this);
	a->setStatusTip(tr("previous 1X"));
	connect(a, SIGNAL(triggered()), this, SLOT(onPrevious1X()));
	myAnimationActions.insert(ANIMATION_PREVIOUS_1X, a);

	a = new QAction(QIcon(":/Resources/ani_playback.png"), tr("&animation back play"), this);
	a->setStatusTip(tr("animation back play"));
	connect(a, SIGNAL(triggered()), this, SLOT(onAnimationPlayBack()));
	myAnimationActions.insert(ANIMATION_PLAY_BACK, a);

	a = new QAction(QIcon(":/Resources/ani_init.png"), tr("&animation initialize"), this);
	a->setStatusTip(tr("animation initialize"));
	connect(a, SIGNAL(triggered()), this, SLOT(onGoFirstStep()));
	myAnimationActions.insert(ANIMATION_INIT, a);

	a = new QAction(QIcon(":/Resources/ani_play.png"), tr("&animation play"), this);
	a->setStatusTip(tr("animation play"));
	connect(a, SIGNAL(triggered()), this, SLOT(onAnimationPlay()));
	myAnimationActions.insert(ANIMATION_PLAY, a);

	a = new QAction(QIcon(":/Resources/ani_fast.png"), tr("&forward1X"), this);
	a->setStatusTip(tr("forward 1X"));
	connect(a, SIGNAL(triggered()), this, SLOT(onForward1X()));
	myAnimationActions.insert(ANIMATION_FORWARD_1X, a);

	a = new QAction(QIcon(":/Resources/ani_morefast.png"), tr("&forward2X"), this);
	a->setStatusTip(tr("forward 2X"));
	connect(a, SIGNAL(triggered()), this, SLOT(onForward2X()));
	myAnimationActions.insert(ANIMATION_FORWARD_2X, a);

	a = new QAction(QIcon(":/Resources/ani_toEnd.png"), tr("&toEnd"), this);
	a->setStatusTip(tr("Go to end"));
	connect(a, SIGNAL(triggered()), this, SLOT(onGoToEnd()));
	myAnimationActions.insert(ANIMATION_GO_END, a);

	for (int i = 0; i < myAnimationActions.size(); i++)
	{
		myAnimationBar->addAction(myAnimationActions.at(i));
	}
	HSlider = new QSlider(Qt::Orientation::Horizontal, this);
	HSlider->setFixedWidth(100);
	connect(HSlider, SIGNAL(valueChanged(int)), this, SLOT(ani_scrollbar()));
	connect(gl, SIGNAL(mySignal()), SLOT(mySlot()));
	myAnimationBar->addWidget(HSlider);

	LEframe = new QLineEdit(this);
	LEframe->setText(QString("0"));
	LEframe->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LEframe->setContentsMargins(QMargins(5, 0, 0, 0));
	myAnimationBar->addWidget(LEframe);
	Lframe = new QLabel(this);
	Lframe->setText(QString("/ 0"));
	Lframe->setContentsMargins(QMargins(5, 0, 0, 0));
	myAnimationBar->addWidget(Lframe);
	LETimes = new QLineEdit(this);
	LETimes->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	LEframe->setFixedWidth(50);
	LETimes->setFixedWidth(50);
	QLabel *LTimes = new QLabel(this);
	LTimes->setText(QString("Time : "));
	LTimes->setContentsMargins(QMargins(10, 0, 0, 0));
	myAnimationBar->addWidget(LTimes);
	myAnimationBar->addWidget(LETimes);
	myAnimationBar->hide();
}

void xdynamics::ChangeComboBox(int id)
{
	gl->ChangeDisplayOption(id);
}

void xdynamics::newproj()
{
	newDialog nd;
	int ret = nd.exec();
	if (ret)
	{
		createToolOperations();
		createAnimationOperations();
		_isOnMainActions = true;
		model::name = nd.name;
		model::path = nd.path + (nd.isBrowser ? "" : nd.name + "/");
		mg = new modelManager;
		db = new database(this, mg);
		addDockWidget(Qt::RightDockWidgetArea, db);
		if (nd.isBrowser)
			mg->OpenModel(nd.fullPath);
		else
		{
			unit_type u = nd.unit;
			gravity_direction dg = nd.dir_g;
			model::setGravity(DEFAULT_GRAVITY, dg);
			model::unit = nd.unit;
		}		
	}
	else
	{
		if (!mg)
			mg = new modelManager;
		db = new database(this, mg);
		addDockWidget(Qt::RightDockWidgetArea, db);
		//mg->openModeler(gl, nd->fullPath());
		//ui.PTEdit->appendPlainText("Open file : " + nd->fullPath());
		if (!_isOnMainActions)
			setMainAction();
		_isOnMainActions = true;
	}
	cmd = new cmdWindow(this);
	addDockWidget(Qt::BottomDockWidgetArea, cmd);
}

// CODEDYN
void xdynamics::openproj()
{
	QString file_path = QFileDialog::getOpenFileName(
		this, tr("open"), model::path,
		tr("Model file (*.xdm);;All files (*.*)"));
	mg->OpenModel(file_path);
}

void xdynamics::setAnimationAction(bool b)
{
	foreach(QAction* a, myAnimationActions)
	{
		a->setEnabled(b);
	}
	myAnimationBar->show();
}

void xdynamics::saveproj()
{
	mg->SaveCurrentModel();
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

void xdynamics::onGoToBegin()
{
	vcontroller::initFrame();
	mySlot();
	onAnimationPause();
	operating_animation = ANIMATION_GO_BEGIN;
	vcontroller::off_play();
	//updateAnimationFrame();
}

void xdynamics::onPrevious2X()
{
	onAnimationPause();
	vcontroller::move2previous2x();
	operating_animation = ANIMATION_PREVIOUS_2X;
	onSetPlayAnimation();
}

void xdynamics::onPrevious1X()
{
	onAnimationPause();
	vcontroller::move2previous1x();
	operating_animation = ANIMATION_PREVIOUS_1X;
	mySlot();
}

void xdynamics::onSetPlayAnimation()
{
	QAction *a = NULL; 
	switch (operating_animation)
	{
	case ANIMATION_PREVIOUS_2X:
		a = myAnimationActions[ANIMATION_PREVIOUS_2X];
		vcontroller::setPlayMode(ANIMATION_PREVIOUS_2X);
		break;
	case ANIMATION_FORWARD_2X:
		a = myAnimationActions[ANIMATION_FORWARD_2X];
		vcontroller::setPlayMode(ANIMATION_FORWARD_2X);
		break;
	case ANIMATION_PLAY:
		a = myAnimationActions[ANIMATION_PLAY];
		vcontroller::setPlayMode(ANIMATION_PLAY);
		break;
	case ANIMATION_PLAY_BACK:
		a = myAnimationActions[ANIMATION_PLAY_BACK];
		vcontroller::setPlayMode(ANIMATION_PLAY_BACK);
		break;
	}
	a->disconnect();
	a->setIcon(QIcon(":/Resources/ani_pause.png"));
	a->setStatusTip(tr("Restart the animation."));
	connect(a, SIGNAL(triggered()), this, SLOT(onAnimationPause()));
	vcontroller::on_play();
}

void xdynamics::onAnimationPlay()
{
	onAnimationPause();
	operating_animation = ANIMATION_PLAY;
	onSetPlayAnimation();
}

void xdynamics::onAnimationPlayBack()
{
	onAnimationPause();
	operating_animation = ANIMATION_PLAY_BACK;
	onSetPlayAnimation();
}

void xdynamics::onAnimationPause()
{
	QAction *a = NULL;
	QString icon_path;
	switch (operating_animation)
	{
	case ANIMATION_PLAY:
		a = myAnimationActions[ANIMATION_PLAY];
		icon_path = ":/Resources/ani_play.png";
		break;
	case ANIMATION_PLAY_BACK:
		a = myAnimationActions[ANIMATION_PLAY_BACK];
		icon_path = ":/Resources/ani_playback.png";
		break;
	case ANIMATION_PREVIOUS_2X:
		a = myAnimationActions[ANIMATION_PREVIOUS_2X];
		icon_path = ":/Resources/ani_moreprevious.png";
		break;
	case ANIMATION_FORWARD_2X:
		a = myAnimationActions[ANIMATION_FORWARD_2X];
		icon_path = ":/Resources/ani_morefast.png";
		break;
	default:
		return;
	}
	if (a)
	{
		a->disconnect();
		a->setIcon(QIcon(icon_path));
		a->setStatusTip(tr("Pause the animation."));
	}	
	switch (operating_animation)
	{
	case ANIMATION_PLAY:
		connect(a, SIGNAL(triggered()), this, SLOT(onAnimationPlay()));
		break;
	case ANIMATION_PLAY_BACK:
		connect(a, SIGNAL(triggered()), this, SLOT(onAnimationPlayBack()));
		break;
	case ANIMATION_PREVIOUS_2X:
		connect(a, SIGNAL(triggered()), this, SLOT(onPrevious2X()));
		break;
	case ANIMATION_FORWARD_2X:
		connect(a, SIGNAL(triggered()), this, SLOT(onForward2X()));
		break;
	}
	vcontroller::off_play();
	//timer->stop();
}

void xdynamics::onForward1X()
{
	onAnimationPause();
	operating_animation = ANIMATION_FORWARD_1X;
	vcontroller::move2forward1x();
	mySlot();
}

void xdynamics::onForward2X()
{
	onAnimationPause();
	operating_animation = ANIMATION_FORWARD_2X;
	onSetPlayAnimation();
}

void xdynamics::onGoToEnd()
{
	vcontroller::moveEnd();
	onAnimationPause();
	operating_animation = ANIMATION_GO_END;
	mySlot();
}

void xdynamics::onGoFirstStep()
{
	vcontroller::moveStart();
	onAnimationPause();
	operating_animation = ANIMATION_GO_END;
	mySlot();
}

void xdynamics::ani_scrollbar()
{
 	if (vcontroller::Play())
 	{
//  		QString tf;
//  		tf.sprintf("%.5f", vcontroller::getTimes());
//  		int frame = vcontroller::getFrame();
//  		HSlider->setValue(frame);
//  		LEframe->setText(QString("%1").arg(frame));
//  		LETimes->setText(tf);
 		return;
 	}
	onAnimationPause();
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
// 	QString dir = md->modelPath();
// 	QString file = QFileDialog::getOpenFileName(this, tr("open"), dir);
// 	if (file.isEmpty())
// 		return;
// 	if (gl->change(file, CHANGE_PARTICLE_POSITION, BIN)){
// 		md->particleSystem()->changeParticlesFromVP(gl->vParticles()->getPosition());
// 	}
}

void xdynamics::ChangeShape()
{

}

void xdynamics::DEMRESULTASCII_Export()
{
// 	ExportDialog edlg(md);
// 	edlg.callDialog();
}

void xdynamics::MBDRESULTASCII_Export()
{
// 	QString dir = md->modelPath();
// 	QString fileName = QFileDialog::getOpenFileName(this, tr("Export"), dir, tr("MBD Result File(*.mrf)"));
// 	QFile qfi(fileName);
// 	QString txtFile = md->modelPath() + "/" + md->modelName() + "_mrf.txt";
// 	qfi.open(QIODevice::ReadOnly);
// 	QFile qfo(txtFile);
// 	qfo.open(QIODevice::WriteOnly);
// 	QTextStream qso(&qfo);
// 
//  	unsigned int cnt = 0;
// 	unsigned int nm = 0;
// 	unsigned int id = 0;
// 	unsigned int nout = 0;
// 	unsigned int name_size = 0;
// 	double ct = 0.0;
// 	char ch;
// 	
// 	VEC3D p, v, a;
// 	EPD ep, ev, ea;
// 	qfi.read((char*)&nm, sizeof(unsigned int));
// 	qfi.read((char*)&nout, sizeof(unsigned int));
// 	while (!qfi.atEnd()){
// 		qfi.read((char*)&cnt, sizeof(unsigned int));
// 		for (unsigned int i = 0; i < nm; i++){
// 			QString str;
// 			qfi.read((char*)&name_size, sizeof(unsigned int));
// 			for (unsigned int j = 0; j < name_size; j++){
// 				qfi.read(&ch, sizeof(char));
// 				str.push_back(ch);
// 			}
// 			qfi.read((char*)&id, sizeof(unsigned int));
// 			qfi.read((char*)&ct, sizeof(double));
// 			qfi.read((char*)&p, sizeof(VEC3D));
// 			qfi.read((char*)&ep, sizeof(EPD));
// 			qfi.read((char*)&v, sizeof(VEC3D));
// 			qfi.read((char*)&ev, sizeof(EPD));
// 			qfi.read((char*)&a, sizeof(VEC3D));
// 			qfi.read((char*)&ea, sizeof(EPD));
// 			qso << ct << " " << str << " " << p.x << " " << p.y << " " << p.z
// 				<< " " << ep.e0 << " " << ep.e1 << " " << ep.e2 << " " << ep.e3
// 				<< " " << v.x << " " << v.y << " " << v.z
// 				<< " " << ev.e0 << " " << ev.e1 << " " << ev.e2 << " " << ev.e3
// 				<< " " << a.x << " " << a.y << " " << a.z
// 				<< " " << ea.e0 << " " << ea.e1 << " " << ea.e2 << " " << ea.e3 << " ";
// 		}
// 		qso << endl;
// 	}
// 
// 	qfo.close();
// 	qfi.close();
}

void xdynamics::MS3DASCII_Import()
{
	importDialog id(this);
	int ret = id.exec();
	if (ret)
	{
		if (!id.file_path.isEmpty())
		{
			int begin = id.file_path.lastIndexOf("/") + 1;
			int end = id.file_path.lastIndexOf(".");
			QString _nm = id.file_path.mid(begin, end - begin);
			vpolygon * vp = gl->makePolygonObject(_nm, MILKSHAPE_3D_ASCII, id.file_path);
			if (!mg->GeometryObject(model::name))
				mg->CreateModel(model::name, modelManager::OBJECTS, true);
			polygonObject* po =
				mg->GeometryObject()->makePolygonObject
				(_nm, BOUNDAR_WALL, id.file_path, MILKSHAPE_3D_ASCII, vp->NumTriangles(), vp->VertexList(), vp->IndexList()
				,(material_type)id.type, id.youngs, id.poisson, id.density, id.shear);
			cmd->printLine();
			cmd->write(CMD_INFO, mg->GeometryObject()->Logs()[po->Name()]);
			cmd->printLine();
		}
	}	
}

void xdynamics::OBJPROPERTY_Dialog()
{
// 	if (!ptObj)
// 		ptObj = new objProperty(this);
// 	ptObj->initialize(md, gl);
// 	ptObj->show();
}

void xdynamics::makeCube()
{
	cubeDialog cd;
	int ret = cd.exec();
	if (ret)
	{
		if (!mg->GeometryObject(model::name))
			mg->CreateModel(model::name, modelManager::OBJECTS, true);
		cube *c = mg->GeometryObject()->makeCube(
			cd.name, (material_type)cd.type, BOUNDAR_WALL, 
			cd.start, cd.end, cd.youngs, cd.poisson, cd.density, cd.shear);
		//c->define(cd.start, cd.end);
		//gl->makeCube(c);
		//db->addChild(database::CUBE_ROOT, c->Name());
		cmd->printLine();
		cmd->write(CMD_INFO, mg->GeometryObject()->Logs()[c->Name()]);
		cmd->printLine();
	}
}

void xdynamics::makePlane()
{
	planeDialog pd;
	int ret = pd.exec();
	if (ret)
	{
		if (!mg->GeometryObject(model::name))
			mg->CreateModel(model::name, modelManager::OBJECTS, true);
		plane *p = mg->GeometryObject()->makePlane(
			pd.name, (material_type)pd.type, BOUNDAR_WALL, 
			pd.Pa, pd.Pb, pd.Pc, pd.Pd, 
			pd.youngs, pd.poisson, pd.density, pd.shear);
		cmd->printLine();
		cmd->write(CMD_INFO, mg->GeometryObject()->Logs()[p->Name()]);
		cmd->printLine();
	}
}

void xdynamics::makePolygon()
{
// 	polygonDialog pd;
// 	gl->makePolygon(pd.callDialog(md));
}

void xdynamics::makeCylinder()
{
// 	cylinderDialog cyd;
// 	gl->makeCylinder(cyd.callDialog(md));
}

void xdynamics::makeLine()
{
	gl->makeLine();
}

void xdynamics::makeParticle()
{
	particleDialog pd;
	int ret = pd.exec();
	if (ret)
	{
		VEC4D* pos;
		if (!mg->DEMModel(model::name))
		{
			mg->CreateModel(model::name, modelManager::DEM, true);
			mg->CreateModel(model::name, modelManager::PARTICLE_MANAGER, true);
		}
		particleManager *pm = mg->DEMModel()->ParticleManager();
		switch (pd.method)
		{
		case 0: pos = pm->CreateCubeParticle
			(pd.name, (material_type)pd.type, pd.ncubex, pd.ncubey, pd.ncubez, 
			 pd.loc[0], pd.loc[1], pd.loc[2], 
			 pd.spacing, pd.min_radius, pd.max_radius,
			 pd.youngs, pd.density, pd.poisson, pd.shear
			 );
			break;
		case 1: pos = pm->CreatePlaneParticle
			(pd.name, (material_type)pd.type, pd.nplanex, pd.nplanez,
			 pd.loc[0], pd.loc[1], pd.loc[2],
			 pd.dir[0], pd.dir[1], pd.dir[2],
			 pd.spacing, pd.min_radius, pd.max_radius,
			 pd.youngs, pd.density, pd.poisson, pd.shear
			 );
			break;
		}
		cmd->printLine();
		cmd->write(CMD_INFO, pm->Logs()[pd.name]);
		cmd->printLine();
		/*gl->makeParticle((double*)pos, pm->Np());*/
	}
}

void xdynamics::makeMass()
{
// 	massDialog msd;
// 	mass* ms = msd.callDialog(md);
// 	if (ms)
// 	{
// 		if (ms->getBaseGeometryType() == POLYGON)
// 			ms->setPolyGeometryObject(gl->getVPolyObjectFromName(ms->name()));
// 		else
// 			ms->setGeometryObject(gl->getVObjectFromName(ms->name()));
// 	}
// 		//ms->setgl->getVObjectFromName(ms->name());
// 		//gl->makeMassCoordinate(ms->name());
}

void xdynamics::makeContactPair()
{
	contactPairDialog cpd;
	QStringList list;
	if (mg->DEMModel()->ParticleManager())
		list.push_back("particles");
	list.append(mg->GeometryObject()->Objects().keys());
	cpd.setObjectLists(list);
	int ret = cpd.exec();
	if (ret)
	{
		if (!mg->ContactManager())
		{
			mg->CreateModel(model::name, modelManager::CONTACT_MANAGER, true);
		}
		particleManager* pm = mg->DEMModel()->ParticleManager();
		geometryObjects* obj = mg->GeometryObject();
		object* o1 = cpd.firstObj == "particles" ? pm->Object() : obj->Object(cpd.firstObj);
		object* o2 = cpd.secondObj == "particles" ? pm->Object() : obj->Object(cpd.secondObj);
		mg->ContactManager()->CreateContactPair
			(
			cpd.name, cpd.method, o1, o2,
			cpd.restitution, cpd.stiffnessRatio, cpd.friction
			);
		cmd->printLine();
		cmd->write(CMD_INFO, mg->ContactManager()->Logs()[cpd.name]);
		cmd->printLine();
	}
//	hmcmDialog hmcm(this, md);
}

void xdynamics::exitThread()
{
	onAnimationPause();
	solver->quit();
	solver->wait();
	solver->disconnect();
	if (solver) delete solver; solver = NULL;
	if (pBar)
	{
		delete pBar;
		pBar = NULL;
	}
// 	if (th->isRunning()){
// 		ui.statusBar->removeWidget(pBar);
// 		delete pBar;
// 		pBar = NULL;
// 
// 		th->exit();
// 		delete solver;
// 		solver = NULL;
// 		simPlayAct->setDisabled(true);
// 	}
}

void xdynamics::recieveProgress(int pt, QString ch, QString info)
{
	if (ch == "__line__")
	{
		cmd->printLine();
		return;
	}
	if (pt >= 0)
	{
		vcontroller::setTotalFrame(pt);
		HSlider->setMaximum(pt);
		QString lf = "/ ";
		QTextStream(&lf) << pt;
		Lframe->setText(lf);
		pBar->setValue(pt);
		cmd->write(CMD_INFO, ch);
	}
	else if (pt == -1)
	{
		cmd->write(CMD_INFO, ch);
	}
}

void xdynamics::deleteFileByEXT(QString ext)
{
	QString dDir = model::path + model::name + "/";
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
// 	disconnect(simPlayAct);
// 	simPlayAct->setIcon(QIcon(":/Resources/ani_play.png"));
// 	simPlayAct->setStatusTip(tr("Restart for simulation."));
// 	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_play()));
// 	//simPlayAct->setEnabled(true);
// 	sim->wait();
}

void xdynamics::solve()
{
	solveDialog sd;
	int ret = sd.exec();
	if (ret <= 0)
		return;
	simulation::dt = sd.time_step;
	simulation::et = sd.sim_time;
	simulation::st = sd.save_step;
	simulation::dev = sd.isCpu ? simulation::CPU : simulation::GPU;

	deleteFileByEXT("bin");
	if (!solver)
		solver = new xDynamicsSolver(mg);
	if (solver->initialize())
	{
		connect(solver, SIGNAL(finishedThread()), this, SLOT(exitThread()));
		connect(solver, SIGNAL(sendProgress(int, QString, QString)), this, SLOT(recieveProgress(int, QString, QString)));
	}
	if (!pBar)
		pBar = new QProgressBar;
	pBar->setMaximum(solver->totalPart());
	//
	ui.statusBar->addWidget(pBar, 1);
	myModelingActions[RUN_ANALYSIS]->disconnect();
	myModelingActions[RUN_ANALYSIS]->setIcon(QIcon(":/Resources/stop.png"));
	myModelingActions[RUN_ANALYSIS]->setStatusTip(tr("Pause for simulation."));
	connect(myModelingActions[RUN_ANALYSIS], SIGNAL(triggered()), solver, SLOT(setStopCondition()));
	myModelingActions[RUN_ANALYSIS]->setEnabled(true);
	saveproj();
	solver->start();
	setAnimationAction(true);
// 	solveDialog sd;
// 	if (!sd.callDialog())
// 		return;
// 
// 	//qDebug() << "Delete *.bin files of " << md->modelPath();
// 	deleteFileByEXT("bin");
// 
// // 	if (th == NULL)
// // 		th = new QThread;
// 	if (!solver)
// 		solver = new xDynamicsSolver(mg);
// // 	if (sim == NULL){
// // 		if (md->numParticle() && !md->numMass()){
// // 			sim = new dem_simulation(md);
// // 		}
// // 		else if (!md->numParticle() && md->numMass()){
// // 			sim = new mbd_simulation(md);
// // 		}
// // 		else if (md->numParticle() && md->numMass()){
// // 			sim = new dembd_simulation(md, new dem_simulation(md), new mbd_simulation(md));
// // 		}
// // 	}
// 	simulation::st = sd.saveStep;
// 	simulation::dt = sd.timeStep;
// 	simulation::et = sd.simTime;
// 	simulation::dev = static_cast<simulation::deviceType>(sd.dev);
// 	connect(solver, SIGNAL(finishedThread()), this, SLOT(exitThread()));
// 	connect(solver, SIGNAL(sendProgress(int, QString, QString)), this, SLOT(recieveProgress(int, QString, QString)));
// 	//sim->setSimulationCondition(sd.simTime, sd.timeStep, sd.saveStep);
// 	if (solver->initialize())
// 	{
// 		//sim->setCommandWindow(cmd);
// 		cmd->write(CMD_INFO, QString("Simulation is initialized."));
// 		//qDebug() << "- Initialization of simulation ---------------------------- DONE";
// 		if (pBar == NULL)
// 			pBar = new QProgressBar;
// 		pBar->setMaximum(simulation::nstep);
// 		ui.statusBar->addWidget(pBar, 1);
// 		//sim->moveToThread(th);
// // 		connect(sim, SIGNAL(finished()), this, SLOT(exitThread()));
// // 		connect(sim, SIGNAL(sendProgress(unsigned int)), this, SLOT(recieveProgress(unsigned int)));
// 		//connect(newAct, SIGNAL(triggered()), this, SLOT(waitSimulation()));
// 	}
// 	else
// 	{
// 		delete solver;
// 		solver = NULL;
// 		return;
// 	}
// 	//md->setSolveDevice((tSolveDevice)!sd.isCpu);
// 	disconnect(simPlayAct);
// 	simPlayAct->setIcon(QIcon(":/Resources/ani_pause.png"));
// 	simPlayAct->setStatusTip(tr("Pause for simulation."));
// 	connect(simPlayAct, SIGNAL(triggered()), this, SLOT(sim_stop()));
// 	simPlayAct->setEnabled(true);
// 	saveproj();
// 	solver->start();
// 	//qDebug() << "- Simulation Thread On (CPU) -";
// 	// 	if (neigh) delete neigh; neigh = NULL;
// 	// 	if (vv) delete vv; vv = NULL;
}

void xdynamics::changePaletteMode()
{
// 	bool iss = gl->changePaletteMode();
// 	if (iss)
// 		paletteAct->setIcon(QIcon(":/Resources/noSketch.png"));
// 	else
// 		paletteAct->setIcon(QIcon(":/Resources/sketch.png"));
}

void xdynamics::changeProjectionViewMode()
{
	projectionType pt = gl->changeProjectionViewMode();
	if (pt == ORTHO_PROJECTION)
		myModelingActions[CHANGE_PROJECTION_VIEW]->setIcon(QIcon(":/Resources/perspective.png"));
	else if (pt == PERSPECTIVE_PROJECTION)
		myModelingActions[CHANGE_PROJECTION_VIEW]->setIcon(QIcon(":/Resources/ortho.png"));
}