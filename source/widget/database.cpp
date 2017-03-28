#include "database.h"
#include "modeler.h"
#include <QMenu>

database::database()
{

}

database::database(QWidget* parent, modeler* _md)
	: QDockWidget(parent)
	, md(_md)
{
	vtree = new QTreeWidget;
	vtree->setColumnCount(1);
	vtree->setHeaderLabel("Database");

	vtree->setContextMenuPolicy(Qt::CustomContextMenu);
	//vtree->setWindowTitle("Database");
	setWidget(vtree);
	roots[PLANE_ROOT] = new QTreeWidgetItem(vtree);
	roots[LINE_ROOT] = new QTreeWidgetItem(vtree);
	roots[CUBE_ROOT] = new QTreeWidgetItem(vtree);
	roots[CYLINDER_ROOT] = new QTreeWidgetItem(vtree);
	roots[POLYGON_ROOT] = new QTreeWidgetItem(vtree);
	roots[MASS_ROOT] = new QTreeWidgetItem(vtree);

	roots[PLANE_ROOT]->setText(0, "Plane");
	roots[LINE_ROOT]->setText(0, "Line");
	roots[CUBE_ROOT]->setText(0, "Cube");
	roots[CYLINDER_ROOT]->setText(0, "Cylinder");
	roots[POLYGON_ROOT]->setText(0, "Polygon");
	roots[MASS_ROOT]->setText(0, "Mass");

	roots[PLANE_ROOT]->setIcon(0, QIcon(":/Resources/icRect.png"));
	roots[LINE_ROOT]->setIcon(0, QIcon(":/Resources/icLine.png"));
	roots[CUBE_ROOT]->setIcon(0, QIcon(":/Resources/pRec.png"));
	roots[CYLINDER_ROOT]->setIcon(0, QIcon(":/Resources/cylinder.png"));
	roots[POLYGON_ROOT]->setIcon(0, QIcon(":/Resources/icPolygon.png"));
	roots[MASS_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));
	connect(vtree, &QTreeWidget::customContextMenuRequested, this, &database::contextMenu);
	md->setDatabase(this);
}

database::~database()
{
	qDeleteAll(roots.begin(), roots.end());
	if (vtree) delete vtree; vtree = NULL;
}

void database::addChild(tRoot tr, QString& _nm)
{
	QTreeWidgetItem* child = new QTreeWidgetItem();
	child->setText(0, _nm);
	roots[tr]->addChild(child);
}

void database::contextMenu(const QPoint& pos)
{
	QTreeWidgetItem* item = vtree->itemAt(pos);
	if (!item->parent())
		return;
	QAction *act = new QAction(tr("Property"), this);
	act->setStatusTip(tr("property menu"));
	connect(act, SIGNAL(triggered()), this, SLOT(actProperty()));

	QMenu menu(this);
	menu.addAction(act);

	QPoint pt(pos);
	menu.exec(vtree->mapToGlobal(pos));
}

void database::actProperty()
{
	//return;
}