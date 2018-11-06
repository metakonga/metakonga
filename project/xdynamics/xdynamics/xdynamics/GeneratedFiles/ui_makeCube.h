/********************************************************************************
** Form generated from reading UI file 'makeCube.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAKECUBE_H
#define UI_MAKECUBE_H

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

class Ui_DLG_MAKECUBE
{
public:
    QGroupBox *GB_MaterialProperty;
    QWidget *widget;
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
    QGroupBox *GB_Information;
    QWidget *widget1;
    QGridLayout *gridLayout_2;
    QLabel *L_Name;
    QLineEdit *LE_Name;
    QLabel *L_StartPoint;
    QLineEdit *LE_StartPoint;
    QLabel *L_EndPoint;
    QLineEdit *LE_EndPoint;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;

    void setupUi(QDialog *DLG_MAKECUBE)
    {
        if (DLG_MAKECUBE->objectName().isEmpty())
            DLG_MAKECUBE->setObjectName(QStringLiteral("DLG_MAKECUBE"));
        DLG_MAKECUBE->resize(320, 323);
        GB_MaterialProperty = new QGroupBox(DLG_MAKECUBE);
        GB_MaterialProperty->setObjectName(QStringLiteral("GB_MaterialProperty"));
        GB_MaterialProperty->setGeometry(QRect(10, 120, 301, 161));
        widget = new QWidget(GB_MaterialProperty);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 21, 281, 126));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_Type = new QLabel(widget);
        L_Type->setObjectName(QStringLiteral("L_Type"));

        gridLayout->addWidget(L_Type, 0, 0, 1, 1);

        CB_Type = new QComboBox(widget);
        CB_Type->setObjectName(QStringLiteral("CB_Type"));

        gridLayout->addWidget(CB_Type, 0, 1, 1, 1);

        L_YoungsModulus = new QLabel(widget);
        L_YoungsModulus->setObjectName(QStringLiteral("L_YoungsModulus"));

        gridLayout->addWidget(L_YoungsModulus, 1, 0, 1, 1);

        LE_Youngs = new QLineEdit(widget);
        LE_Youngs->setObjectName(QStringLiteral("LE_Youngs"));

        gridLayout->addWidget(LE_Youngs, 1, 1, 1, 1);

        L_Density = new QLabel(widget);
        L_Density->setObjectName(QStringLiteral("L_Density"));

        gridLayout->addWidget(L_Density, 2, 0, 1, 1);

        LE_Density = new QLineEdit(widget);
        LE_Density->setObjectName(QStringLiteral("LE_Density"));

        gridLayout->addWidget(LE_Density, 2, 1, 1, 1);

        L_PoissonRatio = new QLabel(widget);
        L_PoissonRatio->setObjectName(QStringLiteral("L_PoissonRatio"));

        gridLayout->addWidget(L_PoissonRatio, 3, 0, 1, 1);

        LE_PoissonRatio = new QLineEdit(widget);
        LE_PoissonRatio->setObjectName(QStringLiteral("LE_PoissonRatio"));

        gridLayout->addWidget(LE_PoissonRatio, 3, 1, 1, 1);

        L_ShearModulus = new QLabel(widget);
        L_ShearModulus->setObjectName(QStringLiteral("L_ShearModulus"));

        gridLayout->addWidget(L_ShearModulus, 4, 0, 1, 1);

        LE_ShearModulus = new QLineEdit(widget);
        LE_ShearModulus->setObjectName(QStringLiteral("LE_ShearModulus"));

        gridLayout->addWidget(LE_ShearModulus, 4, 1, 1, 1);

        GB_Information = new QGroupBox(DLG_MAKECUBE);
        GB_Information->setObjectName(QStringLiteral("GB_Information"));
        GB_Information->setGeometry(QRect(10, 10, 301, 101));
        widget1 = new QWidget(GB_Information);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 20, 281, 74));
        gridLayout_2 = new QGridLayout(widget1);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Name = new QLabel(widget1);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        gridLayout_2->addWidget(L_Name, 0, 0, 1, 1);

        LE_Name = new QLineEdit(widget1);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));

        gridLayout_2->addWidget(LE_Name, 0, 1, 1, 1);

        L_StartPoint = new QLabel(widget1);
        L_StartPoint->setObjectName(QStringLiteral("L_StartPoint"));

        gridLayout_2->addWidget(L_StartPoint, 1, 0, 1, 1);

        LE_StartPoint = new QLineEdit(widget1);
        LE_StartPoint->setObjectName(QStringLiteral("LE_StartPoint"));

        gridLayout_2->addWidget(LE_StartPoint, 1, 1, 1, 1);

        L_EndPoint = new QLabel(widget1);
        L_EndPoint->setObjectName(QStringLiteral("L_EndPoint"));

        gridLayout_2->addWidget(L_EndPoint, 2, 0, 1, 1);

        LE_EndPoint = new QLineEdit(widget1);
        LE_EndPoint->setObjectName(QStringLiteral("LE_EndPoint"));

        gridLayout_2->addWidget(LE_EndPoint, 2, 1, 1, 1);

        widget2 = new QWidget(DLG_MAKECUBE);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(137, 290, 171, 25));
        horizontalLayout = new QHBoxLayout(widget2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(widget2);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(widget2);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        QWidget::setTabOrder(LE_Name, LE_StartPoint);
        QWidget::setTabOrder(LE_StartPoint, LE_EndPoint);
        QWidget::setTabOrder(LE_EndPoint, CB_Type);
        QWidget::setTabOrder(CB_Type, LE_Youngs);
        QWidget::setTabOrder(LE_Youngs, LE_Density);
        QWidget::setTabOrder(LE_Density, LE_PoissonRatio);

        retranslateUi(DLG_MAKECUBE);

        QMetaObject::connectSlotsByName(DLG_MAKECUBE);
    } // setupUi

    void retranslateUi(QDialog *DLG_MAKECUBE)
    {
        DLG_MAKECUBE->setWindowTitle(QApplication::translate("DLG_MAKECUBE", "Make cube object", nullptr));
        GB_MaterialProperty->setTitle(QApplication::translate("DLG_MAKECUBE", "Material property", nullptr));
        L_Type->setText(QApplication::translate("DLG_MAKECUBE", "Type", nullptr));
        L_YoungsModulus->setText(QApplication::translate("DLG_MAKECUBE", "Youngs modulus", nullptr));
        L_Density->setText(QApplication::translate("DLG_MAKECUBE", "Density", nullptr));
        L_PoissonRatio->setText(QApplication::translate("DLG_MAKECUBE", "Poisson ratio", nullptr));
        L_ShearModulus->setText(QApplication::translate("DLG_MAKECUBE", "Shear modulus", nullptr));
        GB_Information->setTitle(QApplication::translate("DLG_MAKECUBE", "Information", nullptr));
        L_Name->setText(QApplication::translate("DLG_MAKECUBE", "Name", nullptr));
        L_StartPoint->setText(QApplication::translate("DLG_MAKECUBE", "Start point", nullptr));
        L_EndPoint->setText(QApplication::translate("DLG_MAKECUBE", "End point", nullptr));
        PB_Ok->setText(QApplication::translate("DLG_MAKECUBE", "Ok", nullptr));
        PB_Cancle->setText(QApplication::translate("DLG_MAKECUBE", "Cancle", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_MAKECUBE: public Ui_DLG_MAKECUBE {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAKECUBE_H
