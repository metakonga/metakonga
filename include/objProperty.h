#ifndef OBJPROPERTY_H
#define OBJPROPERTY_H

#include "ui_objProperty.h"
#include <QDialog>
#include <QMap>

class modeler;
class object;
class GLWidget;


QT_BEGIN_NAMESPACE
class QListWidget;
class QTableWidget;
class QLineEdit;
class QComboBox;
QT_END_NAMESPACE

class objProperty : public QDialog
{
	Q_OBJECT

	enum tobjProperty{ TOP_PARTICLE = 0, TOP_POLYGON_OBJECT, TOP_OBJECT };
public:
	objProperty();
	objProperty(QWidget* parent = NULL);
	~objProperty();

	void initialize(modeler* _md, GLWidget* _gl);
	void settingPolygonObjectProperties(int id);

private:
	QLineEdit* getNameWidget();
	QComboBox* getMaterialWidget();

	bool _isSetting;
	bool _isInit;
	int lastSelection;

	Ui::ObjProperty *ui;

	modeler *md;
	GLWidget *gl;

	QListWidget *objList;
	QTableWidget *objProp;

	QMap<int, tobjProperty> tops;
	tobjProperty currentTargetType;
	//QLineEdit *LEName;
	//QLineEdit *LEMaterial;
	object* currentObject;

private slots:
	void changeObject(int);
	void changeMaterial(int);
};
#endif