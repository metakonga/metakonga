#ifndef MESSAGEBOX_H
#define MESSAGEBOX_H

#include <QMessageBox>
#include <QString>

class messageBox
{
public:
	messageBox();
	~messageBox();

	static void setMessageData(QString text, QString info = "", QMessageBox::StandardButtons buttons = QMessageBox::Default, QMessageBox::StandardButton button = QMessageBox::Default);

	static int run();
	static int run(QString text, QString info = "", QMessageBox::StandardButtons buttons = QMessageBox::Default, QMessageBox::StandardButton button = QMessageBox::Default);
private:
	static QString _text;
	static QString _info;
	static QMessageBox::StandardButtons _buttons;
	static QMessageBox::StandardButton _button;
	static QMessageBox *msg;
};

#endif