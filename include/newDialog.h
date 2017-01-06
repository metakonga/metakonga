#ifndef NEWDIALOG_H
#define NEWDIALOG_H

#include <QDialog>
#include "mphysics_types.h"

namespace Ui{
	class newModelDialog;
}

class newDialog : public QDialog
{
	Q_OBJECT

public:
	explicit newDialog(QWidget *parent = 0);
	~newDialog();

	bool callDialog();
	QString name() { return _name; }
	QString path() { return _path; }
	QString fullPath() { return _fullPath; }
	tUnit unit() { return _unit; }
	tGravity gravityDirection() { return _dir_g; }

private:
// 	QLineEdit *LEName;
// 	QLineEdit *LEPath;

	QString _name;
	QString _path;
	QString _fullPath;

	tUnit _unit;
	tGravity _dir_g;

	bool isDialogOk;

	Ui::newModelDialog *ui;

private slots:
	void Click_ok();
	void Click_browse();
};


#endif