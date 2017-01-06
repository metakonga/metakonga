#ifndef MSGBOX_H
#define MSGBOX_H

#include "QMessageBox.h"

class msgBox : public QMessageBox
{
public:
	msgBox(QString msg, QMessageBox::Icon ic)
	{
		setIcon(ic);
		setText(msg);
		exec();
	}
};

#endif