#include "database.h"
#include "modelManager.h"
#include <QMenu>

database* db;

database::database()
{

}

database::database(QWidget* parent, modelManager* _md)
	: QDockWidget(parent)
	, md(_md)
{
	db = this;
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
	roots[COLLISION_ROOT] = new QTreeWidgetItem(vtree);
	roots[PARTICLES_ROOT] = new QTreeWidgetItem(vtree);

	roots[PLANE_ROOT]->setText(0, "Plane");
	roots[LINE_ROOT]->setText(0, "Line");
	roots[CUBE_ROOT]->setText(0, "Cube");
	roots[CYLINDER_ROOT]->setText(0, "Cylinder");
	roots[POLYGON_ROOT]->setText(0, "Polygon");
	roots[MASS_ROOT]->setText(0, "Mass");
	roots[COLLISION_ROOT]->setText(0, "Collision");
	roots[PARTICLES_ROOT]->setText(0, "Particles");

	roots[PLANE_ROOT]->setIcon(0, QIcon(":/Resources/icRect.png"));
	roots[LINE_ROOT]->setIcon(0, QIcon(":/Resources/icLine.png"));
	roots[CUBE_ROOT]->setIcon(0, QIcon(":/Resources/pRec.png"));
	roots[CYLINDER_ROOT]->setIcon(0, QIcon(":/Resources/cylinder.png"));
	roots[POLYGON_ROOT]->setIcon(0, QIcon(":/Resources/icPolygon.png"));
	roots[MASS_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));
	roots[COLLISION_ROOT]->setIcon(0, QIcon(":/Resources/collision.png"));
	roots[PARTICLES_ROOT]->setIcon(0, QIcon(":/Resources/particle.png"));
	connect(vtree, &QTreeWidget::customContextMenuRequested, this, &database::contextMenu);
	md->setDatabase(this);
}

database::~database()
{
	qDeleteAll(roots.begin(), roots.end());
	if (vtree) delete vtree; vtree = NULL;
}

database* database::DB()
{
	return db;
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
	if (!item)
		return;
	if (!item->parent())
		return;
	//QString c = item->text(0);
	QAction *act0 = new QAction(tr("Property"), this);
	act0->setWhatsThis(item->text(0));
	act0->setStatusTip(tr("property menu"));
	QAction *act1 = new QAction(tr("Delete"), this);
	act1->setStatusTip(tr("delete menu"));
// 	connect(act0, SIGNAL(triggered()), this, SLOT(actProperty()));
// 	connect(act1, SIGNAL(triggered()), this, SLOT(actDelete()));

	QMenu menu(item->text(0), this);
	menu.addAction(act0);
	menu.addAction(act1);

	QPoint pt(pos);
	QAction *a = menu.exec(vtree->mapToGlobal(pos));

	if (a)
	{
		QString txt = a->text();
		if (txt == "Delete")
			modelManager::MM()->ActionDelete(item->text(0));
	}
}

// void database::actProperty()
// {
// 	//return;
// }
// 
// void database::actDelete()
// {
// 	
// }
