#ifndef IMPORTDIALOG_H
#define IMPORTDIALOG_H

#include "ui_import.h"

class importDialog : public QDialog, private Ui::DLG_IMPORT_SHAPE
{
	Q_OBJECT

public:
	importDialog(QWidget* parent = NULL);
	~importDialog();

	QString file_path;
	int type;
	double com[3];
	double youngs;
	double poisson;
	double density;
	double shear;

private slots:
	void click_browser();
	void changeComboBox(int);
	void Click_ok();
	void Click_cancel();
};

#endif