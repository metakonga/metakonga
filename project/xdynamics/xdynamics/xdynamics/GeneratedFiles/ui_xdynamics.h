/********************************************************************************
** Form generated from reading UI file 'xdynamics.ui'
**
** Created by: Qt User Interface Compiler version 5.4.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_XDYNAMICS_H
#define UI_XDYNAMICS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_xdynamics
{
public:
    QAction *actionChange_Shape;
    QAction *actionMilkShape_3D_ASCII;
    QAction *actionMBD_Result_ASCII;
    QAction *actionDEM_Result_ASCII;
    QAction *actionProperty;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QScrollArea *GraphicArea;
    QWidget *scrollAreaWidgetContents;
    QMenuBar *menuBar;
    QMenu *menu;
    QMenu *menuImport;
    QMenu *menuExport;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QToolBar *secToolBar;

	void setupUi(QMainWindow *xdynamics)
    {
		if (xdynamics->objectName().isEmpty())
			xdynamics->setObjectName(QStringLiteral("xdynamics"));
		xdynamics->resize(600, 400);
		actionChange_Shape = new QAction(xdynamics);
        actionChange_Shape->setObjectName(QStringLiteral("actionChange_Shape"));
        actionChange_Shape->setCheckable(true);
		actionMilkShape_3D_ASCII = new QAction(xdynamics);
        actionMilkShape_3D_ASCII->setObjectName(QStringLiteral("actionMilkShape_3D_ASCII"));
		actionMBD_Result_ASCII = new QAction(xdynamics);
        actionMBD_Result_ASCII->setObjectName(QStringLiteral("actionMBD_Result_ASCII"));
		actionDEM_Result_ASCII = new QAction(xdynamics);
        actionDEM_Result_ASCII->setObjectName(QStringLiteral("actionDEM_Result_ASCII"));
		actionProperty = new QAction(xdynamics);
        actionProperty->setObjectName(QStringLiteral("actionProperty"));
		centralWidget = new QWidget(xdynamics);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        GraphicArea = new QScrollArea(centralWidget);
        GraphicArea->setObjectName(QStringLiteral("GraphicArea"));
        GraphicArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 580, 315));
        GraphicArea->setWidget(scrollAreaWidgetContents);

        gridLayout->addWidget(GraphicArea, 0, 0, 1, 1);

		xdynamics->setCentralWidget(centralWidget);
		menuBar = new QMenuBar(xdynamics);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        menu = new QMenu(menuBar);
        menu->setObjectName(QStringLiteral("menu"));
        menuImport = new QMenu(menu);
        menuImport->setObjectName(QStringLiteral("menuImport"));
        menuExport = new QMenu(menu);
        menuExport->setObjectName(QStringLiteral("menuExport"));
		xdynamics->setMenuBar(menuBar);
		mainToolBar = new QToolBar(xdynamics);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        mainToolBar->setMovable(false);
        mainToolBar->setFloatable(false);
		xdynamics->addToolBar(Qt::TopToolBarArea, mainToolBar);
		statusBar = new QStatusBar(xdynamics);
        statusBar->setObjectName(QStringLiteral("statusBar"));
		xdynamics->setStatusBar(statusBar);
		secToolBar = new QToolBar(xdynamics);
        secToolBar->setObjectName(QStringLiteral("secToolBar"));
		xdynamics->addToolBar(Qt::TopToolBarArea, secToolBar);
		xdynamics->insertToolBarBreak(secToolBar);

        menuBar->addAction(menu->menuAction());
        menu->addAction(actionChange_Shape);
        menu->addAction(menuExport->menuAction());
        menu->addAction(menuImport->menuAction());
        menu->addAction(actionProperty);
        menuImport->addAction(actionMilkShape_3D_ASCII);
        menuExport->addAction(actionMBD_Result_ASCII);
        menuExport->addAction(actionDEM_Result_ASCII);

		retranslateUi(xdynamics);

		QMetaObject::connectSlotsByName(xdynamics);
    } // setupUi

	void retranslateUi(QMainWindow *xdynamics)
    {
		xdynamics->setWindowTitle(QApplication::translate("xdynamics", "xdynamics", 0));
        actionChange_Shape->setText(QApplication::translate("xdynamics", "Change Shape", 0));
        actionMilkShape_3D_ASCII->setText(QApplication::translate("xdynamics", "MilkShape 3D ASCII", 0));
        actionMBD_Result_ASCII->setText(QApplication::translate("xdynamics", "MBD Result ASCII", 0));
        actionDEM_Result_ASCII->setText(QApplication::translate("xdynamics", "DEM Result ASCII", 0));
        actionProperty->setText(QApplication::translate("xdynamics", "Property", 0));
        menu->setTitle(QApplication::translate("xdynamics", "\352\270\260\353\212\245", 0));
        menuImport->setTitle(QApplication::translate("xdynamics", "Import", 0));
        menuExport->setTitle(QApplication::translate("xdynamics", "Export", 0));
        secToolBar->setWindowTitle(QApplication::translate("xdynamics", "toolBar", 0));
    } // retranslateUi

};

namespace Ui {
	class xdynamics : public Ui_xdynamics {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_XDYNAMICS_H
