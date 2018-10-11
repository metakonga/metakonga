#ifndef LINEEDITWIDGET_H
#define LINEEDITWIDGET_H

#include <QLineEdit>

class lineEditWidget : public QLineEdit
{
	Q_OBJECT

public:
	lineEditWidget();
	~lineEditWidget();

signals:
	void up_arrow_key_press();

protected:
	virtual void keyPressEvent(QKeyEvent *e);
};

#endif