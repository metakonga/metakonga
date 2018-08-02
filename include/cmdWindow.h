#ifndef CMDWINDOW_H
#define CMDWINDOW_H

#include <QMap>
#include <QDockWidget>
#include <QPlainTextEdit>

enum tWriting{ CMD_INFO = 0, CMD_DEBUG, CMD_ERROR };

class cmdWindow : public QDockWidget
{
	Q_OBJECT

public:
	cmdWindow();
	cmdWindow(QWidget* parent);
	~cmdWindow();

	void write(tWriting tw, QString& c);
	void printLine();
	//void addChild(tRoot, QString& _nm);

private:
	QPlainTextEdit *cmd;
};

#endif