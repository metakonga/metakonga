#ifndef DATABASE_H
#define DATABASE_H

#include "types.h"
#include <QMap>
#include <QDockWidget>
#include <QTreeWidget>

class modelManager;

class database : public QDockWidget
{
	Q_OBJECT
public:
	enum tRoot { PLANE_ROOT = 0, LINE_ROOT, CUBE_ROOT, CYLINDER_ROOT, POLYGON_ROOT, 
		RIGID_BODY_ROOT, COLLISION_ROOT, PARTICLES_ROOT, CONSTRAINT_ROOT, SPRING_DAMPER_ROOT };

public:
	database();
	database(QWidget* parent, modelManager* _md);
	~database();

	static database* DB();

	void addChild(tRoot, QString& _nm);

	private slots:
	void contextMenu(const QPoint&);

signals:
	void contextSignal(QString, context_menu);
// 	void actProperty();
// 	void actDelete();

private:
	QTreeWidget *vtree;
	QMap<tRoot, QTreeWidgetItem*> roots;
	
	modelManager* md;
};

#endif