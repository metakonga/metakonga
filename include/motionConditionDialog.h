#ifndef MOTION_CONDITION_DIALOG_H
#define MOTION_CONDITION_DIALOG_H

#include "ui_motionCondition.h"

class object;

class motionConditionDialog : public QDialog, private Ui::DLG_MotionCondition
{
	Q_OBJECT

public:
	motionConditionDialog(QWidget* parent = NULL);
	//cube(std::map<QString, QObject*> *_objs);
	~motionConditionDialog();

	void setName(QString& nm);

	bool isConstantVelocity;
	double st;
	double et;
	double cv;
	double ux, uy, uz;

	private slots:
	void Click_ok();
	void Click_cancel();
};

#endif