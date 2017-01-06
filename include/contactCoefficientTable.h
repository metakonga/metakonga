#ifndef CONTACTCOEFFICIENTTABLE_H
#define CONTACTCOEFFICIENTTABLE_H

#include <QDialog>
#include <QModelIndex>
#include <map>
#include "mphysics_types.h"

namespace parview{
	class Object;
}

QT_BEGIN_NAMESPACE
class QPushButton;
class QTableWidget;
class QTableWidgetItem;
QT_END_NAMESPACE

class contactCoefficientTable : public QDialog
{
	Q_OBJECT

public:
	contactCoefficientTable(QWidget *parent = 0);
	~contactCoefficientTable();

	void setTable(std::map<parview::Object*, parview::Object*>& pcont);
	//std::map<QString, ccontactConstant>& ContactConstants() { return cconsts; }

	private slots:
	void Click_Ok();
	void Click_Cancel();
	void actionClick(QModelIndex);

private:
	QTableWidget *cctable;
	QPushButton *PBOk;
	QPushButton *PBCancel;
	
	//std::map<QString, ccontactConstant> cconsts;
};

#endif