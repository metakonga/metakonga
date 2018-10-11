#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_xdynamics_car.h"
#include "ComponentTree.h"

class xdynamics_car : public QMainWindow
{
	Q_OBJECT

public:
	xdynamics_car(QWidget *parent = Q_NULLPTR);
	~xdynamics_car();

	private slots:
	void clickModelType();

private:
	Ui::WM_XDYNAMICS_CAR ui;
	ComponentTree *ctree;
};
