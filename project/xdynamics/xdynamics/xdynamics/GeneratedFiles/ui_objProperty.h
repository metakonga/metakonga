/********************************************************************************
** Form generated from reading UI file 'objProperty.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_OBJPROPERTY_H
#define UI_OBJPROPERTY_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ObjProperty
{
public:
    QGridLayout *gridLayout_2;
    QScrollArea *listArea;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout;
    QScrollArea *propertyArea;
    QWidget *scrollAreaWidgetContents_2;

    void setupUi(QDialog *ObjProperty)
    {
        if (ObjProperty->objectName().isEmpty())
            ObjProperty->setObjectName(QStringLiteral("ObjProperty"));
        ObjProperty->resize(400, 443);
        gridLayout_2 = new QGridLayout(ObjProperty);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        listArea = new QScrollArea(ObjProperty);
        listArea->setObjectName(QStringLiteral("listArea"));
        listArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 380, 210));
        gridLayout = new QGridLayout(scrollAreaWidgetContents);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        listArea->setWidget(scrollAreaWidgetContents);

        gridLayout_2->addWidget(listArea, 0, 0, 1, 1);

        propertyArea = new QScrollArea(ObjProperty);
        propertyArea->setObjectName(QStringLiteral("propertyArea"));
        propertyArea->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QStringLiteral("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 380, 205));
        propertyArea->setWidget(scrollAreaWidgetContents_2);

        gridLayout_2->addWidget(propertyArea, 1, 0, 1, 1);


        retranslateUi(ObjProperty);

        QMetaObject::connectSlotsByName(ObjProperty);
    } // setupUi

    void retranslateUi(QDialog *ObjProperty)
    {
        ObjProperty->setWindowTitle(QApplication::translate("ObjProperty", "Properies", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ObjProperty: public Ui_ObjProperty {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_OBJPROPERTY_H
