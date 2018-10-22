#ifndef NEWDIALOG_H
#define NEWDIALOG_H

#include <QDialog>
#include "types.h"
#include "ui_newModel.h"

class newDialog : public QDialog, private Ui::DLG_NewModel
{
	Q_OBJECT

public:
	explicit newDialog(QWidget *parent = 0);
	~newDialog();

	bool isBrowser;
	bool isSinglePrecision;

	QString name;
	QString path;
	QString fullPath;

	unit_type unit;
	gravity_direction dir_g;

private slots:
	void Click_ok();
	void Click_browse();
};


#endif