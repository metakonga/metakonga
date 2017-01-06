#ifndef CUBEDIALOG_H
#define CUBEDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QGridLayout;
class QComboBox;
QT_END_NAMESPACE

class modeler;
class cube;

class cubeDialog : public QDialog
{
	Q_OBJECT

public:
	cubeDialog();
	//cube(std::map<QString, QObject*> *_objs);
	~cubeDialog();
	cube* callDialog(modeler *md);

	bool isDialogOk;
	QLabel *LMaterial;
	QComboBox *CBMaterial;
	QLabel *LStartPoint;
	QLabel *LEndPoint;
	QLabel *LName;
	QLineEdit *LEName;
	QLineEdit *LEStartPoint;
	QLineEdit *LEEndPoint;
	QGridLayout *cubeLayout;
	QPushButton *PBOk;
	QPushButton *PBCancel;

private slots:
	void Click_ok();
	void Click_cancel();
};

#endif