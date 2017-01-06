#ifndef PLANEDIALOG_H
#define PLANEDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QComboBox;
QT_END_NAMESPACE

class modeler;
class plane;

class planeDialog : public QDialog
{
	Q_OBJECT

public:
	planeDialog();
	~planeDialog();

	plane* callDialog(modeler *md);

	bool isDialogOk;
	QLabel *LMaterial;
	QComboBox *CBMaterial;
	QDialog *rectDialog;
	QLineEdit *LEName;
	QLineEdit *LEPa;
	QLineEdit *LEPb;
	QLineEdit *LEPc;
	QLineEdit *LEPd;

private slots:
	void Click_ok();
	void Click_cancel();
};


#endif