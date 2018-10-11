#ifndef XDYNAMICS_H
#define XDYNAMICS_H

#include <QtWidgets/QMainWindow>
#include "ui_xdynamics.h"
#include "particleInfoDialog.h"
#include "glwidget.h"
#include "cmdWindow.h"
#include "modelManager.h"
#include "commandManager.h"
#include "startingModel.h"

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
	enum { MAKE_CUBE=0, MAKE_RECT, MAKE_LINE, MAKE_POLY, MAKE_CYLINDER, MAKE_PARTICLE, MAKE_MASS, MAKE_COLLISION, RUN_ANALYSIS, CHANGE_PROJECTION_VIEW, PRE_DEFINE_MBD };
	enum { ANIMATION_GO_BEGIN = 0, ANIMATION_PREVIOUS_2X, ANIMATION_PREVIOUS_1X, ANIMATION_PLAY_BACK, ANIMATION_INIT, ANIMATION_PLAY, ANIMATION_PAUSE, ANIMATION_FORWARD_2X, ANIMATION_FORWARD_1X, ANIMATION_GO_END };
	
	xdynamics(int argc, char** argv, QWidget *parent = 0);
	~xdynamics();

	void setBaseAction();
	void createMainOperations();
	void createToolOperations();
	void createAnimationOperations();

	private slots:
	void ChangeShape();
	void ChangeParticleFromFile();
	void DEMRESULTASCII_Export();
	void SHAPE_Import();
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
	void ani_scrollbar();
	void makeCube();
	void makePlane();
	void makeLine();
	void makePolygon();
	void makeCylinder();
	void makeParticle();
	void makeMass();
	void makeContactPair();
	void preDefinedMBD();
	void changePaletteMode();
	void changeProjectionViewMode();
	void solve();
	void exitThread();
	//void 
	void recieveProgress(int, QString, QString = "");
	void excuteMessageBox();
	void propertySlot(QString, context_object_type);
	//void messageSlot(QString);
	void editingCommandLine();
	void write_command_line_passed_data();
	void openPinfoDialog();
	void deleteFileByEXT(QString ext);

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
	QDockWidget *comm;
	modelManager *mg;
	commandManager *comMgr;
	startingModel *st_model;
};

#endif // xdynamics_H
