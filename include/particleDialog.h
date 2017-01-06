#ifndef PARTICLEDIALOG_H
#define PARTICLEDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QStringList;
class QPushButton;
class QLineEdit;
class QLabel;
class QComboBox;
class QGridLayout;
class QTabWidget;
QT_END_NAMESPACE

class modeler;
class particle_system;

class particleDialog : public QDialog
{
	Q_OBJECT
public:
	particleDialog();
	~particleDialog();

	particle_system* callDialog(modeler *md);

private:
	//QString baseGeometry;

	bool isDialogOk;
	QLabel *LMaterial;
	QComboBox *CBMaterial;
	QWidget *byGeoTab;
	QWidget *byManualTab;
	QTabWidget *tabWidget;
	QComboBox *CBGeometry;
	QLineEdit *LESpacing;
	QLineEdit *LEMRadius;
	QLineEdit *LEName;
	QLineEdit *LERadius;
	QLineEdit *LEPosition;
	QLineEdit *LEVelocity;
	QLineEdit *LERestitution;
	QLineEdit *LEStiffRatio;
	QLineEdit *LEFriction;
	QLineEdit *LECohesion;
	QLineEdit *LENumParticle;
	QLineEdit *LETotalMass;
	QGridLayout *particleLayout;
	QStringList geoComboxList;
	QStringList cpProcess;

	modeler *md;

private slots:
	void Click_ok();
	void Click_cancel();
	void particleInformation();
};

#endif