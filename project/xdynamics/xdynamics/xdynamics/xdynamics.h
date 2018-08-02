#ifndef XDYNAMICS_H
#define XDYNAMICS_H

#include <QtWidgets/QMainWindow>
#include "ui_xdynamics.h"
#include "particleInfoDialog.h"
#include "glwidget.h"
#include "cmdWindow.h"
#include "modelManager.h"

class QThread;
class QLabel;
class QProgressBar;
class objProperty;

#define PROGRAM_NAME "xDynamics"

class xdynamics : public QMainWindow
{
	Q_OBJECT

public:
	enum { NEW = 0, OPEN, SAVE };
	enum { MAKE_CUBE=0, MAKE_RECT, MAKE_LINE, MAKE_POLY, MAKE_CYLINDER, MAKE_PARTICLE, MAKE_MASS, MAKE_COLLISION, RUN_ANALYSIS, CHANGE_PROJECTION_VIEW };
	enum { ANIMATION_GO_BEGIN = 0, ANIMATION_PREVIOUS_2X, ANIMATION_PREVIOUS_1X, ANIMATION_PLAY_BACK, ANIMATION_INIT, ANIMATION_PLAY, ANIMATION_PAUSE, ANIMATION_FORWARD_2X, ANIMATION_FORWARD_1X, ANIMATION_GO_END };
	
	xdynamics(int argc, char** argv, QWidget *parent = 0);
	~xdynamics();

	void setBaseAction();
	void setMainAction();
	void createMainOperations();
	void createToolOperations();
	void createAnimationOperations();

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
	void onGoToBegin();
	void onPrevious2X();
	void onPrevious1X();
	void onSetPlayAnimation();
	void onAnimationPlay();
	void onAnimationPlayBack();
	void onAnimationPause();
	void onForward1X();
	void onForward2X();
	void onGoToEnd();
	void onGoFirstStep();
// 	void ani_previous2x();
// 	void ani_previous1x();
// 	void ani_play();
// 	void sim_play();
// 	void sim_stop();
// 	void ani_pause();
// 	void ani_forward1x();
// 	void ani_forward2x();
	void ani_scrollbar();
	void makeCube();
	void makePlane();
	void makeLine();
	void makePolygon();
	void makeCylinder();
	void makeParticle();
	void makeMass();
	void makeContactPair();
	void changePaletteMode();
	void changeProjectionViewMode();
	void solve();
	void exitThread();
	void recieveProgress(int, QString, QString = "");

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

	int operating_animation;

	QString previous_directory;

	QToolBar* myModelingBar;
	QToolBar* myAnimationBar;
	QList<QAction*> myMainActions;
	QList<QAction*> myModelingActions;
	QList<QAction*> myAnimationActions;

	//QAction *MenuChangeShapeAct;
// 	QAction *newAct;
// 	QAction *openAct;
// 	QAction *openRtAct;
// 	QAction *saveAct;
// 	QAction *simPlayAct;
// 	QAction *simStopAct;
// 	QAction *aniPlayAct;
// 	QAction *aniPauseAct;
// 	QAction *aniPreviousAct;
// 	QAction *aniPreviousAct2;
// 	QAction *aniForwardAct;
// 	QAction *aniForwardAct2;
// 	QAction *makeCubeAct;
// 	QAction *makeLineAct;
// 	QAction *makeRectAct;
// 	QAction *makePolyAct;
// 	QAction *makeCylinderAct;
// 	QAction *makeParticleAct;
// 	QAction *makeMassAct;
// 	QAction *paletteAct;
// 	QAction *projectionViewAct;
// 	QAction *collidConstAct;
// 	QAction *solveProcessAct;
// 	QAction *changeParticleAct;

	//QAction *pinfoAct;

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

	modelManager *mg;
};

#endif // xdynamics_H
