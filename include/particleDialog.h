#ifndef PARTICLEDIALOG_H
#define PARTICLEDIALOG_H

 #include "ui_makeParticle.h"
// 
// class modeler;
// 
class particleDialog : public QDialog, private Ui::DLG_MakeParticle
{
	Q_OBJECT

public:
	particleDialog(QWidget* parent = NULL);
	~particleDialog();

	QString name;

	int type;
	int method;
	bool real_time;
	unsigned int perNp;
	unsigned int np;
	double circle_diameter;
	double spacing;
	double min_radius;
	double max_radius;
	unsigned int ncubex, nplanex;
	unsigned int ncubey, nplanez;
	unsigned int ncubez;
	double dir[3];
	double loc[3];
	double stiffnessRatio;
	double youngs;
	double poisson;
	double density;
	double shear;

private:
	void setCubeData();
	void setPlaneData();
	void setCircleData();

private slots:
	void changeComboBox(int);
	void changeTab(int);
	void click_ok();
	void click_cancle();
	void update_tnp();
};

#endif
// 	Q_OBJECT
// 
// public:
// 	particleDialog(QWidget* parent, modeler* md);
// 	~particleDialog();
// 
// private:
// 	modeler* md;
// 
// 	void setupDialog();
// 
// 	private slots:
// 
// 	void check_stack_particle();
// 	void change_particle_radius();
// 	void change_stack_number();
// 	//void check_stiffnessRatio();
// 	void click_ok();
// 	void click_cancle();
// };
// 
// #endif
// 
// // #ifndef PARTICLEDIALOG_H
// // #define PARTICLEDIALOG_H
// // 
// // #include <QDialog>
// // 
// // QT_BEGIN_NAMESPACE
// // class QStringList;
// // class QPushButton;
// // class QLineEdit;
// // class QLabel;
// // class QComboBox;
// // class QGridLayout;
// // class QTabWidget;
// // QT_END_NAMESPACE
// // 
// // class modeler;
// // class particle_system;
// // 
// // class particleDialog : public QDialog
// // {
// // 	Q_OBJECT
// // public:
// // 	particleDialog();
// // 	~particleDialog();
// // 
// // 	particle_system* callDialog(modeler *md);
// // 
// // private:
// // 	//QString baseGeometry;
// // 
// // 	bool isDialogOk;
// // 	QLabel *LMaterial;
// // 	QComboBox *CBMaterial;
// // 	QWidget *byGeoTab;
// // 	QWidget *byManualTab;
// // 	QTabWidget *tabWidget;
// // 	QComboBox *CBGeometry;
// // 	QLineEdit *LESpacing;
// // 	QLineEdit *LEMRadius;
// // 	QLineEdit *LEName;
// // 	QLineEdit *LERadius;
// // 	QLineEdit *LEPosition;
// // 	QLineEdit *LEVelocity;
// // 	QLineEdit *LERestitution;
// // 	//QLineEdit *LEShearModulus;
// // 	QLineEdit *LEFriction;
// // 	QLineEdit *LERollingFriction;
// // 	QLineEdit *LECohesion;
// // 	QLineEdit *LENumParticle;
// // 	QLineEdit *LETotalMass;
// // 	QGridLayout *particleLayout;
// // 	QStringList geoComboxList;
// // 	QStringList cpProcess;
// // 
// // 	modeler *md;
// // 
// // private slots:
// // 	void Click_ok();
// // 	void Click_cancel();
// // 	void particleInformation();
// // };
// // 
// // #endif