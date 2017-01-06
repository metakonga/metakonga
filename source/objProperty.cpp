#include "objProperty.h"
#include "modeler.h"
#include "glwidget.h"
#include "polygonObject.h"
#include <QListWidget>
#include <QTableWidget>
#include <QLineEdit>
#include <QLabel>
#include <QString>
#include <QComboBox>

objProperty::objProperty()
	: QDialog(NULL)
	, md(NULL)
	, gl(NULL)
	, objList(NULL)
	, objProp(NULL)
	, lastSelection(0)
	, _isInit(false)
	, _isSetting(false)
{

}

objProperty::objProperty(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::ObjProperty)
	, md(NULL)
	, gl(NULL)
	, objList(NULL)
	, objProp(NULL)
	, lastSelection(0)
	, _isInit(false)
	, _isSetting(false)
{
	ui->setupUi(this);
	if (!objList)
		objList = new QListWidget;
	if (!objProp)
		objProp = new QTableWidget;
// 	if (!LEName)
// 		LEName = new QLineEdit;
// 	if (!LEMaterial)
// 		LEMaterial = new QLineEdit;
	
	//objProp->setHorizontalHeaderItem(0, new QTableWidgetItem("Property"));
	//objProp->setHorizontalHeaderItem(1, new QTableWidgetItem("Value"));
	ui->listArea->setWidget(objList);
	ui->propertyArea->setWidget(objProp);
	objProp->setColumnCount(2);
	objProp->verticalHeader()->setVisible(false);
	//objProp->setWordWrap(true);
	objProp->setShowGrid(true);
	objProp->setRowCount(0);
	objProp->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeMode::Stretch);
	objProp->horizontalHeader()->setAutoFillBackground(true);
	connect(objList, SIGNAL(currentRowChanged(int)), this, SLOT(changeObject(int)));
}

objProperty::~objProperty()
{
	if (ui) delete ui; ui = NULL;
	if (objList) delete objList; objList = NULL;
	if (objProp) delete objProp; objProp = NULL;
	//bool b = LEName->isEnabled();
	//if (LEName->isEnabled()) delete LEName; LEName = NULL;
	//if (LEMaterial) delete LEMaterial; LEMaterial = NULL;
}

void objProperty::initialize(modeler* _md, GLWidget* _gl)
{
	md = _md;
	gl = _gl;
	objList->clear();
	tops.clear();
	//objProp->clear();
	
	if (md->objPolygon().size()){
		//objList->addItem("polygon object"));
		foreach(polygonObject value, md->objPolygon())
		{
			tops[objList->count()] = TOP_POLYGON_OBJECT;
			objList->addItem(value.objectName());			
		}
	}
	if (_isInit){
		getNameWidget()->setText("");
		getMaterialWidget()->setCurrentIndex(0);
	}
	else{
		QStringList header;
		header << "Property" << "Value";
		objProp->setHorizontalHeaderLabels(header);
		//QListWidgetItem *it = objList->item(id);
		//polygonObject* po = &(md->objPolygon()[it->text()]);
		QLabel *LName = new QLabel("name");
		QLineEdit* LEName = new QLineEdit;
		//LEName->setText(po->objectName());
		LEName->setReadOnly(true);

		QLabel *LMaterial = new QLabel("Material");
		QComboBox *CBMaterial = new QComboBox;
		CBMaterial->addItems(getMaterialList());

		objProp->setRowCount(1); objProp->setCellWidget(0, 0, LName); objProp->setCellWidget(0, 1, LEName);
		objProp->setRowCount(2); objProp->setCellWidget(1, 0, LMaterial); objProp->setCellWidget(1, 1, CBMaterial);

		connect(CBMaterial, SIGNAL(currentIndexChanged(int)), this, SLOT(changeMaterial(int)));
		_isInit = true;
	}
	
	//connect(LEName, SIGNAL(editingFinished()), this, SLOT(changeName()));
	//currentObject = po;
}

void objProperty::changeObject(int id)
{
	currentTargetType = tops[id];
	//objProp->
	switch (currentTargetType){
	case TOP_POLYGON_OBJECT:
		settingPolygonObjectProperties(id);
		break;
	}
	lastSelection = id;
}

void objProperty::changeMaterial(int id)
{
	if (!_isSetting)
		return;
	tMaterial _tm = tMaterial(id);
	
// 	QLineEdit *LEName = NULL;
	switch (currentTargetType){
	case TOP_POLYGON_OBJECT:
	case TOP_OBJECT:
		currentObject->setMaterial(_tm);
		break;
	}
// 		QWidget* w = objProp->cellWidget(0, 1);
// 		LEName = qobject_cast<QLineEdit*>(w);
// 	}
// 	break;
// 	}
// 	if (LEName)
// 	{
// 		md->objPolygon().find(currentObject->objectName());
// 		md->objPolygon().
// 	}
		//currentObject->setObjectName(LEName->text());
}

QLineEdit* objProperty::getNameWidget()
{
	return qobject_cast<QLineEdit*>(objProp->cellWidget(0, 1));
}

QComboBox* objProperty::getMaterialWidget()
{
	return qobject_cast<QComboBox*>(objProp->cellWidget(1, 1));
}

void objProperty::settingPolygonObjectProperties(int id)
{
	_isSetting = false;
	QListWidgetItem *it = objList->item(id);
	polygonObject* po = &(md->objPolygon()[it->text()]);
	currentObject = po;
	//QLineEdit* LEName = new QLineEdit;
	getNameWidget()->setText(po->objectName());
	//LEName->setReadOnly(true);

	getMaterialWidget()->setCurrentIndex((int)po->materialType());

	_isSetting = true;
}