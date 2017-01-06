#ifndef PARVIEW_H
#define PARVIEW_H

#include <QtWidgets/QMainWindow>
#include "ui_parview.h"
#include "particleInfoDialog.h"
#include "glwidget.h"
#include "DEM/DemSimulation.h"

#include "modeler.h"

class QThread;
class QLabel;
//class solveProcess;
class simulation;
class objProperty;

namespace parview
{
	class parVIEW : public QMainWindow
	{
		Q_OBJECT

	public:
		parVIEW(int argc, char** argv, QWidget *parent = 0);
		~parVIEW();

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
		///void spSlot();
		void newproj();
		void openproj();
		void openrtproj();
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
		void collidConst();
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
		Ui::parVIEW ui;
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
		//DemSimulation *dem;
		simulation *sim;
		modeler *md;			// 
	};
}

#endif // PARVIEW_H
