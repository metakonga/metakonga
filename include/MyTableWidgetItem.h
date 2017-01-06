#ifndef MYTABLEWIDGETITEM_H
#define MYTABLEWIDGETITEM_H

#include <QTablewidgetItem>

class MyTableWidgetItem : public QTableWidgetItem
{
public:
	MyTableWidgetItem(QString str) : QTableWidgetItem(str) {}
	~MyTableWidgetItem(){}

	bool operator <(const QTableWidgetItem &other) const
	{
		return text().toDouble() < other.text().toDouble();
	}
};

#endif