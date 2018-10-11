#include "xdynamics_car.h"
#include "ComponentTree.h"
#include <QFile>

xdynamics_car::xdynamics_car(QWidget *parent)
	: QMainWindow(parent)
	, ctree(NULL)
{
	ui.setupUi(this);
	QFrame *frame = new QFrame(ui.ModelingTab->widget(0));
	QStringList headers;
	headers << tr("Component");
	ctree = new ComponentTree(headers, frame);
	QGridLayout *layout = new QGridLayout(ui.ModelingTab->widget(0));
	layout->setMargin(0);
	layout->addWidget(ctree);

	connect(ui.RB_FullCar, SIGNAL(clicked()), this, SLOT(clickModelType()));
	connect(ui.RB_QuarterCar, SIGNAL(clicked()), this, SLOT(clickModelType()));
	connect(ui.RB_TestBed, SIGNAL(clicked()), this, SLOT(clickModelType()));
}

xdynamics_car::~xdynamics_car()
{

}

void xdynamics_car::clickModelType()
{
	if (ui.RB_FullCar->isChecked())
	{
		ctree->setFullCarComponent();
	}
	else if (ui.RB_QuarterCar->isChecked())
	{
		ctree->setQuarterCarComponent();
	}
	else
	{
		ctree->setTestBedComponent();
	}
}
