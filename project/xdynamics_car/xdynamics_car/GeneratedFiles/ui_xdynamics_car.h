/********************************************************************************
** Form generated from reading UI file 'xdynamics_cargtOVlZ.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef XDYNAMICS_CARGTOVLZ_H
#define XDYNAMICS_CARGTOVLZ_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_WM_XDYNAMICS_CAR
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QScrollArea *GraphicArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *RightWidgetLayout;
    QGroupBox *GB_Apply_Suspension;
    QWidget *widget;
    QHBoxLayout *CheckBoxLayout;
    QRadioButton *RB_FullCar;
    QRadioButton *RB_QuarterCar;
    QRadioButton *RB_TestBed;
    QTabWidget *ModelingTab;
    QWidget *CarModeling_Tab;
    QWidget *RoadModeling_Tab;
    QWidget *Simulation_Tab;
    QFrame *frame;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *WM_XDYNAMICS_CAR)
    {
        if (WM_XDYNAMICS_CAR->objectName().isEmpty())
            WM_XDYNAMICS_CAR->setObjectName(QStringLiteral("WM_XDYNAMICS_CAR"));
        WM_XDYNAMICS_CAR->resize(879, 734);
        centralWidget = new QWidget(WM_XDYNAMICS_CAR);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        GraphicArea = new QScrollArea(centralWidget);
        GraphicArea->setObjectName(QStringLiteral("GraphicArea"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(GraphicArea->sizePolicy().hasHeightForWidth());
        GraphicArea->setSizePolicy(sizePolicy);
        GraphicArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 466, 661));
        GraphicArea->setWidget(scrollAreaWidgetContents);

        horizontalLayout->addWidget(GraphicArea);

        RightWidgetLayout = new QVBoxLayout();
        RightWidgetLayout->setSpacing(6);
        RightWidgetLayout->setObjectName(QStringLiteral("RightWidgetLayout"));
        GB_Apply_Suspension = new QGroupBox(centralWidget);
        GB_Apply_Suspension->setObjectName(QStringLiteral("GB_Apply_Suspension"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(GB_Apply_Suspension->sizePolicy().hasHeightForWidth());
        GB_Apply_Suspension->setSizePolicy(sizePolicy1);
        GB_Apply_Suspension->setMinimumSize(QSize(385, 50));
        GB_Apply_Suspension->setMaximumSize(QSize(385, 50));
        widget = new QWidget(GB_Apply_Suspension);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(11, 20, 361, 18));
        CheckBoxLayout = new QHBoxLayout(widget);
        CheckBoxLayout->setSpacing(6);
        CheckBoxLayout->setContentsMargins(11, 11, 11, 11);
        CheckBoxLayout->setObjectName(QStringLiteral("CheckBoxLayout"));
        CheckBoxLayout->setContentsMargins(0, 0, 0, 0);
        RB_FullCar = new QRadioButton(widget);
        RB_FullCar->setObjectName(QStringLiteral("RB_FullCar"));
        RB_FullCar->setChecked(true);

        CheckBoxLayout->addWidget(RB_FullCar);

        RB_QuarterCar = new QRadioButton(widget);
        RB_QuarterCar->setObjectName(QStringLiteral("RB_QuarterCar"));

        CheckBoxLayout->addWidget(RB_QuarterCar);

        RB_TestBed = new QRadioButton(widget);
        RB_TestBed->setObjectName(QStringLiteral("RB_TestBed"));

        CheckBoxLayout->addWidget(RB_TestBed);


        RightWidgetLayout->addWidget(GB_Apply_Suspension);

        ModelingTab = new QTabWidget(centralWidget);
        ModelingTab->setObjectName(QStringLiteral("ModelingTab"));
        QSizePolicy sizePolicy2(QSizePolicy::Fixed, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(ModelingTab->sizePolicy().hasHeightForWidth());
        ModelingTab->setSizePolicy(sizePolicy2);
        ModelingTab->setMinimumSize(QSize(385, 0));
        ModelingTab->setMaximumSize(QSize(385, 16777215));
        CarModeling_Tab = new QWidget();
        CarModeling_Tab->setObjectName(QStringLiteral("CarModeling_Tab"));
        ModelingTab->addTab(CarModeling_Tab, QString());
        RoadModeling_Tab = new QWidget();
        RoadModeling_Tab->setObjectName(QStringLiteral("RoadModeling_Tab"));
        ModelingTab->addTab(RoadModeling_Tab, QString());
        Simulation_Tab = new QWidget();
        Simulation_Tab->setObjectName(QStringLiteral("Simulation_Tab"));
        ModelingTab->addTab(Simulation_Tab, QString());

        RightWidgetLayout->addWidget(ModelingTab);

        frame = new QFrame(centralWidget);
        frame->setObjectName(QStringLiteral("frame"));
        sizePolicy1.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy1);
        frame->setMinimumSize(QSize(385, 250));
        frame->setMaximumSize(QSize(385, 16777215));
        frame->setFrameShape(QFrame::Box);
        frame->setFrameShadow(QFrame::Raised);

        RightWidgetLayout->addWidget(frame);


        horizontalLayout->addLayout(RightWidgetLayout);

        WM_XDYNAMICS_CAR->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(WM_XDYNAMICS_CAR);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 879, 21));
        WM_XDYNAMICS_CAR->setMenuBar(menuBar);
        mainToolBar = new QToolBar(WM_XDYNAMICS_CAR);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        WM_XDYNAMICS_CAR->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(WM_XDYNAMICS_CAR);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        WM_XDYNAMICS_CAR->setStatusBar(statusBar);

        retranslateUi(WM_XDYNAMICS_CAR);

        ModelingTab->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(WM_XDYNAMICS_CAR);
    } // setupUi

    void retranslateUi(QMainWindow *WM_XDYNAMICS_CAR)
    {
        WM_XDYNAMICS_CAR->setWindowTitle(QApplication::translate("WM_XDYNAMICS_CAR", "xdynamics car", nullptr));
        GB_Apply_Suspension->setTitle(QApplication::translate("WM_XDYNAMICS_CAR", "Apply Suspension", nullptr));
        RB_FullCar->setText(QApplication::translate("WM_XDYNAMICS_CAR", "Full Car", nullptr));
        RB_QuarterCar->setText(QApplication::translate("WM_XDYNAMICS_CAR", "Quarter Car", nullptr));
        RB_TestBed->setText(QApplication::translate("WM_XDYNAMICS_CAR", "Test bed", nullptr));
        ModelingTab->setTabText(ModelingTab->indexOf(CarModeling_Tab), QApplication::translate("WM_XDYNAMICS_CAR", "Car Modeling", nullptr));
        ModelingTab->setTabText(ModelingTab->indexOf(RoadModeling_Tab), QApplication::translate("WM_XDYNAMICS_CAR", "Road Modeling", nullptr));
        ModelingTab->setTabText(ModelingTab->indexOf(Simulation_Tab), QApplication::translate("WM_XDYNAMICS_CAR", "Simulation", nullptr));
    } // retranslateUi

};

namespace Ui {
    class WM_XDYNAMICS_CAR: public Ui_WM_XDYNAMICS_CAR {};
} // namespace Ui

QT_END_NAMESPACE

#endif // XDYNAMICS_CARGTOVLZ_H
