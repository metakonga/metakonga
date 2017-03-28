#include "database.h"

database::database()
{

}

database::database(QWidget* parent, modeler* _md)
	: QDockWidget(parent)
	, md(_md)
{
	vtree = new QTreeView;
	setWidget(vtree);
	setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
	setWindowTitle("database");
	
}

database::~database()
{

}