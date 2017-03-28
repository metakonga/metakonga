#ifndef DATABASE_H
#define DATABASE_H

#include <QMap>
#include <QDockWidget>
#include <QTreeWidget>

class modeler;

class database : public QDockWidget
{
	Q_OBJECT
public:
	enum tRoot { PLANE_ROOT = 0, LINE_ROOT, CUBE_ROOT, CYLINDER_ROOT, POLYGON_ROOT, MASS_ROOT };

public:
	database();
	database(QWidget* parent, modeler* _md);
	~database();

	void addChild(tRoot, QString& _nm);

	private slots:
	void contextMenu(const QPoint&);
	void actProperty();

private:
	QTreeWidget *vtree;
	QMap<tRoot, QTreeWidgetItem*> roots;
	
	modeler* md;
};

#endif