#include "cmdWindow.h"

cmdWindow::cmdWindow()
{

}

cmdWindow::cmdWindow(QWidget* parent)
	: QDockWidget(parent)
{
	cmd = new QPlainTextEdit;
	setWidget(cmd);
	cmd->setReadOnly(true);
}

cmdWindow::~cmdWindow()
{
	if (cmd) delete cmd; cmd = NULL;
}

void cmdWindow::write(tWriting tw, QString c)
{
	QString t;
	switch (tw)
	{
	case CMD_INFO: t = ""; break;
	case CMD_DEBUG: t = "* "; break;
	case CMD_ERROR: t = "? "; break;
	case CMD_QUESTION: t = "!"; break;
	}
	c.prepend(t);
	//QString cc = cmd->toPlainText();
	cmd->appendPlainText(c);
	//c.clear();
}

void cmdWindow::printLine()
{
	cmd->appendPlainText("\n-------------------------------------------------------------------------------\n");
}