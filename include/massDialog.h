#ifndef MASSDIALOG_H
#define MASSDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QPushButton;
class QGridLayout;
class QComboBox;
QT_END_NAMESPACE

class modeler;
class mass;

class massDialog : public QDialog
{
	Q_OBJECT

public:
	massDialog();
	//cube(std::map<QString, QObject*> *_objs);
	~massDialog();
	mass* callDialog(modeler *md);

	bool isDialogOk;
	QPushButton *PBOk;
	QPushButton *PBCancel;

	private slots:
	void Click_ok();
	void Click_cancel();
};

#endif