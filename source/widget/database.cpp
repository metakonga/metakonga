#include "database.h"
#include "modelManager.h"
#include "glwidget.h"
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
	roots[RIGID_BODY_ROOT] = new QTreeWidgetItem(vtree);
	roots[COLLISION_ROOT] = new QTreeWidgetItem(vtree);
	roots[PARTICLES_ROOT] = new QTreeWidgetItem(vtree);
	roots[CONSTRAINT_ROOT] = new QTreeWidgetItem(vtree);
	roots[SPRING_DAMPER_ROOT] = new QTreeWidgetItem(vtree);

	roots[PLANE_ROOT]->setText(0, "Plane");
	roots[LINE_ROOT]->setText(0, "Line");
	roots[CUBE_ROOT]->setText(0, "Cube");
	roots[CYLINDER_ROOT]->setText(0, "Cylinder");
	roots[POLYGON_ROOT]->setText(0, "Polygon");
	roots[RIGID_BODY_ROOT]->setText(0, "Mass");
	roots[COLLISION_ROOT]->setText(0, "Collision");
	roots[PARTICLES_ROOT]->setText(0, "Particles");
	roots[CONSTRAINT_ROOT]->setText(0, "Constraint");
	roots[SPRING_DAMPER_ROOT]->setText(0, "SpringDamper");

	roots[PLANE_ROOT]->setIcon(0, QIcon(":/Resources/icRect.png"));
	roots[LINE_ROOT]->setIcon(0, QIcon(":/Resources/icLine.png"));
	roots[CUBE_ROOT]->setIcon(0, QIcon(":/Resources/pRec.png"));
	roots[CYLINDER_ROOT]->setIcon(0, QIcon(":/Resources/cylinder.png"));
	roots[POLYGON_ROOT]->setIcon(0, QIcon(":/Resources/icPolygon.png"));
	roots[RIGID_BODY_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));
	roots[COLLISION_ROOT]->setIcon(0, QIcon(":/Resources/collision.png"));
	roots[PARTICLES_ROOT]->setIcon(0, QIcon(":/Resources/particle.png"));
	roots[CONSTRAINT_ROOT]->setIcon(0, QIcon(":/Resources/spherical.png"));
	roots[SPRING_DAMPER_ROOT]->setIcon(0, QIcon(":/Resources/TSDA_icon.png"));
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
	int col = vtree->currentColumn();
	if (!item)
		return;
	if (!item->parent())
		return;
	QString it = item->text(0);
	QMenu menu(item->text(0), this);
	menu.addAction("Delete");
	menu.addAction("Property");

	QPoint pt(pos);
	QAction *a = menu.exec(vtree->mapToGlobal(pos));

	if (a)
	{
		QString txt = a->text();
		if (txt == "Delete")
		{
			QTreeWidgetItem* parent = item->parent();
 			modelManager::MM()->ActionDelete(item->text(0));
 			GLWidget::GLObject()->actionDelete(item->text(0));
 			parent->removeChild(item);
			delete item;
		}			
	}
	menu.clear();
//	qDeleteAll(menu);
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
