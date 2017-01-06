#include "contactCoefficientTable.h"
#include "MyTableWidgetItem.h"
#include "Object.h"
#include <QtWidgets>

contactCoefficientTable::contactCoefficientTable(QWidget *parent) : QDialog(parent)
{
	cctable = new QTableWidget(0, 5);
	//QTableWidget cctable(pairContact.size(), 5);
	cctable->setSelectionBehavior(QAbstractItemView::SelectRows);
	QStringList labels;
	labels << tr("First Obj.") << tr("Second Obj.") << tr("Friction") << tr("Restitution coef.") << tr("Ratio(ks/kn)");
	cctable->setHorizontalHeaderLabels(labels);
	cctable->verticalHeader()->hide();
	cctable->setShowGrid(true);
	
	PBOk = new QPushButton("Ok");
	PBCancel = new QPushButton("Cancel");
	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_Ok()));
	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_Cancel()));
	QGridLayout *mainLayout = new QGridLayout;
	mainLayout->setSizeConstraint(QLayout::SetNoConstraint);
	mainLayout->addWidget(cctable, 0, 0, 1, 5);
	mainLayout->addWidget(PBOk, 1, 3, 1, 1);
	mainLayout->addWidget(PBCancel, 1, 4, 1, 1);
	setLayout(mainLayout);
	setWindowTitle("Input Contact Coefficient Table");
	resize(700, 300);
	setWindowModality(Qt::WindowModality::ApplicationModal);
	//ccWidget.show();
}

contactCoefficientTable::~contactCoefficientTable()
{

}

void contactCoefficientTable::setTable(std::map<parview::Object*, parview::Object*>& pcont)
{
// 	cctable->setSortingEnabled(false);
// 	cctable->clearContents();
// 	cctable->setRowCount(0);
// 	QString str;
// 	int cnt = 0;
// 	std::map<parview::Object*, parview::Object*>::iterator it;
// 	for (it = pcont.begin(); it != pcont.end(); it++){
// 		cctable->insertRow(cnt);
// 		QTextStream(&str) << it->first->Name();  MyTableWidgetItem *Obj1 = new MyTableWidgetItem(str); str.clear();
// 		QTextStream(&str) << it->second->Name();  MyTableWidgetItem *Obj2 = new MyTableWidgetItem(str); str.clear();
// 		cctable->setItem(cnt, 0, Obj1);
// 		cctable->setItem(cnt, 1, Obj2);
// 	}
}

void contactCoefficientTable::Click_Ok()
{
// 	ccontactConstant cc;
// 	QTableWidgetItem *item;
// 	for (int i = 0; i < cctable->rowCount(); i++){
// 		item = cctable->item(i, 2); cc.friction = item->text().toDouble();
// 		item = cctable->item(i, 3); cc.restitution = item->text().toDouble();
// 		item = cctable->item(i, 4); cc.ratio = item->text().toDouble();
// 		item = cctable->item(i, 0); 
// 		cconsts[item->text()] = cc;
// 	}
}

void contactCoefficientTable::Click_Cancel()
{

}

void contactCoefficientTable::actionClick(QModelIndex index)
{
	int row = index.row();
	int col = index.column();

}

