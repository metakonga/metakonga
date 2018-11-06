/********************************************************************************
** Form generated from reading UI file 'solve.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SOLVE_H
#define UI_SOLVE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_SOLVE
{
public:
    QWidget *widget;
    QGridLayout *gridLayout;
    QLabel *L_TimeStep;
    QLineEdit *LE_TimeStep;
    QLabel *L_SaveStep;
    QLineEdit *LE_SaveStep;
    QLabel *L_SimulationTime;
    QLineEdit *LE_SimulationTime;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *RB_CPU;
    QRadioButton *RB_GPU;

    void setupUi(QDialog *DLG_SOLVE)
    {
        if (DLG_SOLVE->objectName().isEmpty())
            DLG_SOLVE->setObjectName(QStringLiteral("DLG_SOLVE"));
        DLG_SOLVE->resize(267, 164);
        widget = new QWidget(DLG_SOLVE);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 50, 245, 74));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_TimeStep = new QLabel(widget);
        L_TimeStep->setObjectName(QStringLiteral("L_TimeStep"));

        gridLayout->addWidget(L_TimeStep, 0, 0, 1, 1);

        LE_TimeStep = new QLineEdit(widget);
        LE_TimeStep->setObjectName(QStringLiteral("LE_TimeStep"));

        gridLayout->addWidget(LE_TimeStep, 0, 1, 1, 1);

        L_SaveStep = new QLabel(widget);
        L_SaveStep->setObjectName(QStringLiteral("L_SaveStep"));

        gridLayout->addWidget(L_SaveStep, 1, 0, 1, 1);

        LE_SaveStep = new QLineEdit(widget);
        LE_SaveStep->setObjectName(QStringLiteral("LE_SaveStep"));

        gridLayout->addWidget(LE_SaveStep, 1, 1, 1, 1);

        L_SimulationTime = new QLabel(widget);
        L_SimulationTime->setObjectName(QStringLiteral("L_SimulationTime"));

        gridLayout->addWidget(L_SimulationTime, 2, 0, 1, 1);

        LE_SimulationTime = new QLineEdit(widget);
        LE_SimulationTime->setObjectName(QStringLiteral("LE_SimulationTime"));

        gridLayout->addWidget(LE_SimulationTime, 2, 1, 1, 1);

        widget1 = new QWidget(DLG_SOLVE);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(90, 130, 158, 25));
        horizontalLayout = new QHBoxLayout(widget1);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget1);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget1);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        widget2 = new QWidget(DLG_SOLVE);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(10, 20, 241, 18));
        horizontalLayout_2 = new QHBoxLayout(widget2);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        RB_CPU = new QRadioButton(widget2);
        RB_CPU->setObjectName(QStringLiteral("RB_CPU"));

        horizontalLayout_2->addWidget(RB_CPU);

        RB_GPU = new QRadioButton(widget2);
        RB_GPU->setObjectName(QStringLiteral("RB_GPU"));

        horizontalLayout_2->addWidget(RB_GPU);


        retranslateUi(DLG_SOLVE);

        QMetaObject::connectSlotsByName(DLG_SOLVE);
    } // setupUi

    void retranslateUi(QDialog *DLG_SOLVE)
    {
        DLG_SOLVE->setWindowTitle(QApplication::translate("DLG_SOLVE", "Simulation", nullptr));
        L_TimeStep->setText(QApplication::translate("DLG_SOLVE", "Time step", nullptr));
        L_SaveStep->setText(QApplication::translate("DLG_SOLVE", "Save step", nullptr));
        L_SimulationTime->setText(QApplication::translate("DLG_SOLVE", "Simulation time", nullptr));
        PB_Ok->setText(QApplication::translate("DLG_SOLVE", "Ok", nullptr));
        PB_Cancle->setText(QApplication::translate("DLG_SOLVE", "Cancle", nullptr));
        RB_CPU->setText(QApplication::translate("DLG_SOLVE", "CPU Process", nullptr));
        RB_GPU->setText(QApplication::translate("DLG_SOLVE", "GPU Process", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_SOLVE: public Ui_DLG_SOLVE {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SOLVE_H
