#ifndef PLANEDIALOG_H
#define PLANEDIALOG_H

#include "ui_makePlane.h"
#include "vectorTypes.h"

class planeDialog : public QDialog, private Ui::DLG_MAKEPLANE
{
	Q_OBJECT

public:
	planeDialog(QWidget* parent = NULL);
	~planeDialog();

	QString name;
	int type;
	double youngs;
	double poisson;
	double density;
	double shear;
	VEC3D Pa;
	VEC3D Pb;
	VEC3D Pc;
	VEC3D Pd;

private slots:
	void changeComboBox(int);
	void Click_ok();
	void Click_cancel();
};


#endif