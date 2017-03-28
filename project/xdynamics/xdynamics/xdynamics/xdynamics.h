#ifndef XDYNAMICS_H
#define XDYNAMICS_H

#include <QtWidgets/QMainWindow>
#include "ui_xdynamics.h"
#include "particleInfoDialog.h"
#include "glwidget.h"
//#include "DEM/DemSimulation.h"
#include "database.h"
#include "cmdWindow.h"
#include "modeler.h"

class QThread;
class QLabel;
//class solveProcess;
class QProgressBar;
class simulation;
class objProperty;

class xdynamics : public QMainWindow
{
	Q_OBJECT

public:
	xdynamics(int argc, char** argv, QWidget *parent = 0);
	~xdynamics();

	void setBaseAction();
	void setMainAction();

	private slots:
	void ChangeShape();
	void ChangeParticleFromFile();
	void DEMRESULTASCII_Export();
	void MS3DASCII_Import();
	void MBDRESULTASCII_Export();
	void OBJPROPERTY_Dialog();
	void ChangeComboBox(int);
	void mySlot();
	void newproj();
	void openproj();
	void saveproj();
	void ani_previous2x();
	void ani_previous1x();
	void ani_play();
	void sim_play();
	void sim_stop();
	void ani_pause();
	void ani_forward1x();
	void ani_forward2x();
	void ani_scrollbar();
	void makeCube();
	void makePlane();
	void makeLine();
	void makePolygon();
	void makeCylinder();
	void makeParticle();
	void makeMass();
	void makeHMCM();
	void changePaletteMode();
	void changeProjectionViewMode();
	void solve();
	void exitThread();
	void recieveProgress(unsigned int);

	void openPinfoDialog();
	void deleteFileByEXT(QString ext);
	void waitSimulation();

private:
	void setAnimationAction(bool b);

	GLWidget *gl;
	objProperty *ptObj;
	//particleInfoDialog *pinfoDialog;
	Ui::xdynamics ui;
	bool animation_statement;

	bool _isOnMainActions;
	QString previous_directory;

	//QAction *MenuChangeShapeAct;
	QAction *newAct;
	QAction *openAct;
	QAction *openRtAct;
	QAction *saveAct;
	QAction *simPlayAct;
	QAction *simStopAct;
	QAction *aniPlayAct;
	QAction *aniPauseAct;
	QAction *aniPreviousAct;
	QAction *aniPreviousAct2;
	QAction *aniForwardAct;
	QAction *aniForwardAct2;
	QAction *makeCubeAct;
	QAction *makeLineAct;
	QAction *makeRectAct;
	QAction *makePolyAct;
	QAction *makeCylinderAct;
	QAction *makeParticleAct;
	QAction *makeMassAct;
	QAction *paletteAct;
	QAction *projectionViewAct;
	QAction *collidConstAct;
	QAction *solveProcessAct;
	QAction *changeParticleAct;

	QAction *pinfoAct;

	QLabel *Lframe;
	QSlider *HSlider;
	QSlider *PHSlider;
	QLineEdit *LEframe;
	QLineEdit *LETimes;
	QLineEdit *LEparticleID;

	QLabel *LparticleID;
	QComboBox *viewObjectComboBox;

	QProgressBar *pBar;
	QLineEdit *durationTime;

	//solveProcess *sp;
	QThread *th;
	database *db;
	cmdWindow *cmd;
	simulation *sim;
	modeler *md;			// 
};

#endif // xdynamics_H
