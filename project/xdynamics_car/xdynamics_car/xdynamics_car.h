#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_xdynamics_car.h"
#include "ComponentTree.h"
#include "glwidget.h"

class xdynamics_car : public QMainWindow
{
	Q_OBJECT

public:
	xdynamics_car(int argc, char** argv, QWidget *parent = Q_NULLPTR);
	~xdynamics_car();

	private slots:
	void clickModelType();

private:
	Ui::WM_XDYNAMICS_CAR ui;
	ComponentTree *ctree;

	GLWidget *gl;
};
