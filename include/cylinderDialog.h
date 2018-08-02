#ifndef CYLINDERDIALOG
#define CYLINDERDIALOG

#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QGridLayout;
class QRadioButton;
class QComboBox;
QT_END_NAMESPACE

class modeler;
class cylinder;

class cylinderDialog : public QDialog
{
	Q_OBJECT

public:
	cylinderDialog();
	//cube(std::map<QString, QObject*> *_objs);
	~cylinderDialog();
// 	cylinder* callDialog(modeler *md);
// 
// 	bool isDialogOk;
// 	//QLabel *LMaterial;
// 	//QComboBox *CBMaterial;
// 	QLineEdit *LEName;
// 	QLineEdit *LEBaseRadius;
// 	QLineEdit *LETopRadius;
// 	//QLineEdit *LELength;
// 	QLineEdit *LEBasePos;
// 	QLineEdit *LETopPos;
// 	//QGridLayout *cubeLayout;
// 	//QPushButton *PBOk;
// 	//QPushButton *PBCancel;

// 	private slots:
// 	void Click_ok();
// 	void Click_cancel();
};

#endif