#ifndef BODYINFODIALOG_H
#define BODYINFODIALOG_H

#include "ui_bodyInfo.h"

class object;
class pointMass;

class bodyInfoDialog : public QDialog, private Ui::DLG_BODYINFO
{
	Q_OBJECT

public:
	bodyInfoDialog(QWidget* parent = NULL);
	//cube(std::map<QString, QObject*> *_objs);
	~bodyInfoDialog();

	pointMass* setBodyInfomation(object* pm);
	int mt;
	/*double y, d, p;*/
	double mass;
	double volume;
	double x, y, z;
	double ixx, iyy, izz, ixy, iyz, izx;
	double dx, dy, dz;
	double sx, sy, sz;

private slots:
	void Click_ok();
	void Click_cancel();
	void changeMaterialInputType(int);
	void changeMaterialType(int);
};

#endif