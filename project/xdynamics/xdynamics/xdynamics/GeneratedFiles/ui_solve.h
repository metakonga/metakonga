/********************************************************************************
** Form generated from reading UI file 'solveHqukCe.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef SOLVEHQUKCE_H
#define SOLVEHQUKCE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
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
    QWidget *layoutWidget;
    QGridLayout *gridLayout;
    QLabel *L_TimeStep;
    QLineEdit *LE_TimeStep;
    QLabel *L_SaveStep;
    QLineEdit *LE_SaveStep;
    QLabel *L_SimulationTime;
    QLineEdit *LE_SimulationTime;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *layoutWidget2;
    QHBoxLayout *horizontalLayout_2;
    QRadioButton *RB_CPU;
    QRadioButton *RB_GPU;
    QComboBox *CB_MBD_Integrator;
    QComboBox *CB_DEM_Integrator;
    QLabel *label;
    QLabel *label_2;

    void setupUi(QDialog *DLG_SOLVE)
    {
        if (DLG_SOLVE->objectName().isEmpty())
            DLG_SOLVE->setObjectName(QStringLiteral("DLG_SOLVE"));
        DLG_SOLVE->resize(267, 254);
        layoutWidget = new QWidget(DLG_SOLVE);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 140, 245, 74));
        gridLayout = new QGridLayout(layoutWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_TimeStep = new QLabel(layoutWidget);
        L_TimeStep->setObjectName(QStringLiteral("L_TimeStep"));

        gridLayout->addWidget(L_TimeStep, 0, 0, 1, 1);

        LE_TimeStep = new QLineEdit(layoutWidget);
        LE_TimeStep->setObjectName(QStringLiteral("LE_TimeStep"));

        gridLayout->addWidget(LE_TimeStep, 0, 1, 1, 1);

        L_SaveStep = new QLabel(layoutWidget);
        L_SaveStep->setObjectName(QStringLiteral("L_SaveStep"));

        gridLayout->addWidget(L_SaveStep, 1, 0, 1, 1);

        LE_SaveStep = new QLineEdit(layoutWidget);
        LE_SaveStep->setObjectName(QStringLiteral("LE_SaveStep"));

        gridLayout->addWidget(LE_SaveStep, 1, 1, 1, 1);

        L_SimulationTime = new QLabel(layoutWidget);
        L_SimulationTime->setObjectName(QStringLiteral("L_SimulationTime"));

        gridLayout->addWidget(L_SimulationTime, 2, 0, 1, 1);

        LE_SimulationTime = new QLineEdit(layoutWidget);
        LE_SimulationTime->setObjectName(QStringLiteral("LE_SimulationTime"));

        gridLayout->addWidget(LE_SimulationTime, 2, 1, 1, 1);

        layoutWidget1 = new QWidget(DLG_SOLVE);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(90, 220, 158, 25));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(layoutWidget1);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(layoutWidget1);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        layoutWidget2 = new QWidget(DLG_SOLVE);
        layoutWidget2->setObjectName(QStringLiteral("layoutWidget2"));
        layoutWidget2->setGeometry(QRect(10, 10, 241, 18));
        horizontalLayout_2 = new QHBoxLayout(layoutWidget2);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        RB_CPU = new QRadioButton(layoutWidget2);
        RB_CPU->setObjectName(QStringLiteral("RB_CPU"));

        horizontalLayout_2->addWidget(RB_CPU);

        RB_GPU = new QRadioButton(layoutWidget2);
        RB_GPU->setObjectName(QStringLiteral("RB_GPU"));

        horizontalLayout_2->addWidget(RB_GPU);

        CB_MBD_Integrator = new QComboBox(DLG_SOLVE);
        CB_MBD_Integrator->addItem(QString());
        CB_MBD_Integrator->addItem(QString());
        CB_MBD_Integrator->setObjectName(QStringLiteral("CB_MBD_Integrator"));
        CB_MBD_Integrator->setGeometry(QRect(10, 110, 241, 22));
        CB_DEM_Integrator = new QComboBox(DLG_SOLVE);
        CB_DEM_Integrator->addItem(QString());
        CB_DEM_Integrator->setObjectName(QStringLiteral("CB_DEM_Integrator"));
        CB_DEM_Integrator->setGeometry(QRect(10, 60, 241, 22));
        label = new QLabel(DLG_SOLVE);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 40, 91, 16));
        label_2 = new QLabel(DLG_SOLVE);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 90, 91, 16));

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
        CB_MBD_Integrator->setItemText(0, QApplication::translate("DLG_SOLVE", "Implicit HHT-alpha", nullptr));
        CB_MBD_Integrator->setItemText(1, QApplication::translate("DLG_SOLVE", "Explicit Runke-Kutta 2th Order(Nystrom Methods)", nullptr));

        CB_DEM_Integrator->setItemText(0, QApplication::translate("DLG_SOLVE", "Explicit Velocity-Verlet", nullptr));

        label->setText(QApplication::translate("DLG_SOLVE", "DEM Integrator", nullptr));
        label_2->setText(QApplication::translate("DLG_SOLVE", "MBD Integrator", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_SOLVE: public Ui_DLG_SOLVE {};
} // namespace Ui

QT_END_NAMESPACE

#endif // SOLVEHQUKCE_H
