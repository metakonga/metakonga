#include "preDefinedMBDDialog.h"

preDefinedMBDDialog::preDefinedMBDDialog(QWidget* parent /* = NULL */)
	: QDialog(parent)
{
	setupUi(this);
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(click_ok()));
	connect(PB_Cancle, SIGNAL(clicked()), this, SLOT(click_cancle()));
}

preDefinedMBDDialog::~preDefinedMBDDialog()
{
	//qDeleteAll(titems);
}

void preDefinedMBDDialog::setupPreDefinedMBDList(QStringList& mbd_list)
{
	TW_PreDefined_MBD->setColumnCount(1);
	TW_PreDefined_MBD->setHeaderLabels(QStringList() << tr("pre-defined mbd model"));
//	unsigned int idx = 1;
	nitems = mbd_list.size();
	foreach(QString s, mbd_list)
	{
		QTreeWidgetItem *item = new QTreeWidgetItem(TW_PreDefined_MBD);
		item->setText(0, s);
		item->setCheckState(0, Qt::Unchecked);
		//titems.push_back(item);
		//TW_PreDefined_MBD->insertTopLevelItem(item);
	}
}

void preDefinedMBDDialog::click_ok()
{
	for (unsigned int i = 0; i < nitems; i++)
	{
		QTreeWidgetItem* checked_item = TW_PreDefined_MBD->topLevelItem(i);
		if (checked_item->checkState(0))
			checked_items.push_back(checked_item->text(0));
	}
	
	
	//QTreeWidgetItem *checked_item = TW_PreDefined_MBD->itemAt(0, 0);
	
	this->close();
	this->setResult(QDialog::Accepted);
}

void preDefinedMBDDialog::click_cancle()
{
	this->close();
	this->setResult(QDialog::Rejected);
}

