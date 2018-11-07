#ifndef SOLVEDIALOG_H
#define SOLVEDIALOG_H

#include "ui_solve.h"

class solveDialog : public QDialog, private Ui::DLG_SOLVE
{
	Q_OBJECT
public:
	solveDialog(QWidget* parent = NULL);
	~solveDialog();

	bool isCpu;
	int dem_itor_type;
	int mbd_itor_type;
	double time_step;
	unsigned int save_step;
	double sim_time;

	private slots:
	void click_ok();
	void click_cancle();
};

#endif
// 
// #include <QDialog>
// //#include "simulation.h"
// 
// QT_BEGIN_NAMESPACE
// class QLabel;
// class QLineEdit;
// class QRadioButton;
// class QGridLayout;
// class QPushButton;
// QT_END_NAMESPACE
// 
// //class GLWidget;
// class solveDialog : public QDialog
// {
// 	Q_OBJECT
// 
// public:
// 	solveDialog();
// 	///solveDialog(GLWidget *gl);
// 	~solveDialog();
// 
// 	bool callDialog();
// 	int dev;
// 	double simTime;
// 	double timeStep;
// 	unsigned int saveStep;
// 
// 	//	QString caseName;
// 	//		QString basePath;
// 
// private:
// 	//		QDialog *solveDlg;
// 
// 	// 		QLineEdit *LEWorldOrigin;
// 	// 		QLineEdit *LEGridSize;
// 	//		QLineEdit *LECaseName;
// 	//		QLineEdit *LEBasePath;
// 	QRadioButton *RBCpu;
// 	QRadioButton *RBGpu;
// 	QLineEdit *LESimTime;
// 	QLineEdit *LETimeStep;
// 	QLineEdit *LESaveStep;
// 	QGridLayout *solveLayout;
// 	QPushButton *PBSolve;
// 	QPushButton *PBCancel;
// 
// 	// 		vector3<float> worldOrigin;
// 	// 		vector3<unsigned int> gridSize;
// 
// 	//		GLWidget *GL;
// 
// 	bool isDialogOk;
// 
// private slots:
// 	void Click_Solve();
// 	void Click_Cancel();
// };
// 
// 
// #endif