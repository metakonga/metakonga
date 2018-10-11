#include "lineEditWidget.h"

lineEditWidget::lineEditWidget()
{

}

lineEditWidget::~lineEditWidget()
{

}

void lineEditWidget::keyPressEvent(QKeyEvent *e)
{
	QLineEdit::keyPressEvent(e);
	emit(up_arrow_key_press());
}