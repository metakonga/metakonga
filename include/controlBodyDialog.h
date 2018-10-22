#ifndef CONTROLBODYTOOL_H
#define CONTROLBODYTOOL_H

#include "ui_controlBody.h"

class object;
class pointMass;

class controlBodyTool : public QDialog, private Ui::DLG_ControlBody
{
	Q_OBJECT

public:
	controlBodyTool(pointMass* pm, QWidget* parent = NULL);
	//cube(std::map<QString, QObject*> *_objs);
	~controlBodyTool();

private:
	pointMass* t;

	private slots:
	void up();
	void bottom();
	void left();
	void right();
	void xRot();
	void yRot();
	void zRot();
};

#endif