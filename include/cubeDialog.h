#ifndef CUBEDIALOG_H
#define CUBEDIALOG_H

#include "ui_makeCube.h"
#include "types.h"

class cubeDialog : public QDialog, private Ui::DLG_MAKECUBE
{
	Q_OBJECT

public:
	cubeDialog(QWidget* parent = NULL);
	~cubeDialog();

	QString name;
	int type;
	double youngs;
	double poisson;
	double density;
	double shear;
	VEC3D start;
	VEC3D end;

private slots:
	void changeComboBox(int);
	void Click_ok();
	void Click_cancel();
};

#endif