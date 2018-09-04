#ifndef PRE_DEFINED_MBD_DIALOG_H
#define PRE_DEFINED_MBD_DIALOG_H

#include <QDialog>
#include "ui_preDefinedMBD.h"

//class sph_model;

class preDefinedMBDDialog : public QDialog, private Ui::DLG_PREDEFINEDMBD
{
	Q_OBJECT

public:
	preDefinedMBDDialog(QWidget* parent = NULL);
	~preDefinedMBDDialog();

	void setupPreDefinedMBDList(QStringList& mbd_list);

	unsigned int nitems;
	QString target;
	QStringList checked_items;

	QList<QTreeWidgetItem*> titems;

	private slots:
	void click_ok();
	void click_cancle();
};


#endif