/********************************************************************************
** Form generated from reading UI file 'importp16468.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef IMPORTP16468_H
#define IMPORTP16468_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_IMPORT_SHAPE
{
public:
    QGroupBox *GB_MaterialProperty;
    QWidget *layoutWidget_3;
    QGridLayout *gridLayout;
    QLabel *L_Type;
    QComboBox *CB_Type;
    QLabel *L_YoungsModulus;
    QLineEdit *LE_Youngs;
    QLabel *L_Density;
    QLineEdit *LE_Density;
    QLabel *L_PoissonRatio;
    QLineEdit *LE_PoissonRatio;
    QLabel *L_ShearModulus;
    QLineEdit *LE_ShearModulus;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout;
    QLineEdit *LE_FilePath;
    QPushButton *PB_FileBrowser;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_3;
    QLabel *L_CenterOfMass;
    QLineEdit *LE_CenterOfMass;

    void setupUi(QDialog *DLG_IMPORT_SHAPE)
    {
        if (DLG_IMPORT_SHAPE->objectName().isEmpty())
            DLG_IMPORT_SHAPE->setObjectName(QStringLiteral("DLG_IMPORT_SHAPE"));
        DLG_IMPORT_SHAPE->resize(317, 283);
        GB_MaterialProperty = new QGroupBox(DLG_IMPORT_SHAPE);
        GB_MaterialProperty->setObjectName(QStringLiteral("GB_MaterialProperty"));
        GB_MaterialProperty->setGeometry(QRect(10, 80, 301, 161));
        layoutWidget_3 = new QWidget(GB_MaterialProperty);
        layoutWidget_3->setObjectName(QStringLiteral("layoutWidget_3"));
        layoutWidget_3->setGeometry(QRect(10, 21, 281, 126));
        gridLayout = new QGridLayout(layoutWidget_3);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_Type = new QLabel(layoutWidget_3);
        L_Type->setObjectName(QStringLiteral("L_Type"));

        gridLayout->addWidget(L_Type, 0, 0, 1, 1);

        CB_Type = new QComboBox(layoutWidget_3);
        CB_Type->setObjectName(QStringLiteral("CB_Type"));

        gridLayout->addWidget(CB_Type, 0, 1, 1, 1);

        L_YoungsModulus = new QLabel(layoutWidget_3);
        L_YoungsModulus->setObjectName(QStringLiteral("L_YoungsModulus"));

        gridLayout->addWidget(L_YoungsModulus, 1, 0, 1, 1);

        LE_Youngs = new QLineEdit(layoutWidget_3);
        LE_Youngs->setObjectName(QStringLiteral("LE_Youngs"));

        gridLayout->addWidget(LE_Youngs, 1, 1, 1, 1);

        L_Density = new QLabel(layoutWidget_3);
        L_Density->setObjectName(QStringLiteral("L_Density"));

        gridLayout->addWidget(L_Density, 2, 0, 1, 1);

        LE_Density = new QLineEdit(layoutWidget_3);
        LE_Density->setObjectName(QStringLiteral("LE_Density"));

        gridLayout->addWidget(LE_Density, 2, 1, 1, 1);

        L_PoissonRatio = new QLabel(layoutWidget_3);
        L_PoissonRatio->setObjectName(QStringLiteral("L_PoissonRatio"));

        gridLayout->addWidget(L_PoissonRatio, 3, 0, 1, 1);

        LE_PoissonRatio = new QLineEdit(layoutWidget_3);
        LE_PoissonRatio->setObjectName(QStringLiteral("LE_PoissonRatio"));

        gridLayout->addWidget(LE_PoissonRatio, 3, 1, 1, 1);

        L_ShearModulus = new QLabel(layoutWidget_3);
        L_ShearModulus->setObjectName(QStringLiteral("L_ShearModulus"));

        gridLayout->addWidget(L_ShearModulus, 4, 0, 1, 1);

        LE_ShearModulus = new QLineEdit(layoutWidget_3);
        LE_ShearModulus->setObjectName(QStringLiteral("LE_ShearModulus"));

        gridLayout->addWidget(LE_ShearModulus, 4, 1, 1, 1);

        layoutWidget = new QWidget(DLG_IMPORT_SHAPE);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(140, 250, 171, 25));
        horizontalLayout_2 = new QHBoxLayout(layoutWidget);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(layoutWidget);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout_2->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(layoutWidget);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout_2->addWidget(PB_Cancle);

        layoutWidget1 = new QWidget(DLG_IMPORT_SHAPE);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 10, 301, 25));
        horizontalLayout = new QHBoxLayout(layoutWidget1);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        LE_FilePath = new QLineEdit(layoutWidget1);
        LE_FilePath->setObjectName(QStringLiteral("LE_FilePath"));

        horizontalLayout->addWidget(LE_FilePath);

        PB_FileBrowser = new QPushButton(layoutWidget1);
        PB_FileBrowser->setObjectName(QStringLiteral("PB_FileBrowser"));
        PB_FileBrowser->setMaximumSize(QSize(31, 31));

        horizontalLayout->addWidget(PB_FileBrowser);

        widget = new QWidget(DLG_IMPORT_SHAPE);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 44, 301, 22));
        horizontalLayout_3 = new QHBoxLayout(widget);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        L_CenterOfMass = new QLabel(widget);
        L_CenterOfMass->setObjectName(QStringLiteral("L_CenterOfMass"));

        horizontalLayout_3->addWidget(L_CenterOfMass);

        LE_CenterOfMass = new QLineEdit(widget);
        LE_CenterOfMass->setObjectName(QStringLiteral("LE_CenterOfMass"));

        horizontalLayout_3->addWidget(LE_CenterOfMass);


        retranslateUi(DLG_IMPORT_SHAPE);

        QMetaObject::connectSlotsByName(DLG_IMPORT_SHAPE);
    } // setupUi

    void retranslateUi(QDialog *DLG_IMPORT_SHAPE)
    {
        DLG_IMPORT_SHAPE->setWindowTitle(QApplication::translate("DLG_IMPORT_SHAPE", "Import", Q_NULLPTR));
        GB_MaterialProperty->setTitle(QApplication::translate("DLG_IMPORT_SHAPE", "Material property", Q_NULLPTR));
        L_Type->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Type", Q_NULLPTR));
        L_YoungsModulus->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Youngs modulus", Q_NULLPTR));
        L_Density->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Density", Q_NULLPTR));
        L_PoissonRatio->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Poisson ratio", Q_NULLPTR));
        L_ShearModulus->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Shear modulus", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Cancle", Q_NULLPTR));
        PB_FileBrowser->setText(QApplication::translate("DLG_IMPORT_SHAPE", "<<", Q_NULLPTR));
        L_CenterOfMass->setText(QApplication::translate("DLG_IMPORT_SHAPE", "Center of mass", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_IMPORT_SHAPE: public Ui_DLG_IMPORT_SHAPE {};
} // namespace Ui

QT_END_NAMESPACE

#endif // IMPORTP16468_H
