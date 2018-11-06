#include "messageBox.h"

QMessageBox *messageBox::msg = NULL;
QString messageBox::_text = "";
QString messageBox::_info = "";
QMessageBox::StandardButtons messageBox::_buttons = QMessageBox::Default;
QMessageBox::StandardButton messageBox::_button = QMessageBox::Default;

messageBox::messageBox()
{

}

messageBox::~messageBox()
{

}

void messageBox::setMessageData(QString text, QString info /*= ""*/, QMessageBox::StandardButtons buttons /*= QMessageBox::Default*/, QMessageBox::StandardButton button /*= QMessageBox::Default*/)
{
	_text = text;
	_info = info;
	_buttons = buttons;
	_button = button;
}

int messageBox::run(QString text, QString info /* = "" */, QMessageBox::StandardButtons buttons /* = QMessageBox::Default */, QMessageBox::StandardButton button /* = QMessageBox::Default */)
{
	msg = new QMessageBox;
	msg->setText(text);
	if (info != "")
		msg->setInformativeText(info);
	if (buttons != QMessageBox::Default)
		msg->setStandardButtons(buttons);
	if (button != QMessageBox::Default)
		msg->setDefaultButton(button);
	int ret = msg->exec();
	delete msg;
	return ret;
}

int messageBox::run()
{
	msg = new QMessageBox;
	msg->setText(_text);
	if (_info != "")
		msg->setInformativeText(_info);
	if (_buttons != QMessageBox::Default)
		msg->setStandardButtons(_buttons);
	if (_button != QMessageBox::Default)
		msg->setDefaultButton(_button);
	int ret = msg->exec();
	delete msg;
	return ret;
}
