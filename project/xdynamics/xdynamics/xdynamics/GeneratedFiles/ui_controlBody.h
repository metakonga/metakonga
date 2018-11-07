/********************************************************************************
** Form generated from reading UI file 'controlBody.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONTROLBODY_H
#define UI_CONTROLBODY_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QToolButton>

QT_BEGIN_NAMESPACE

class Ui_DLG_ControlBody
{
public:
    QToolButton *TB_UPArrow;
    QToolButton *TB_LEFTArrow;
    QToolButton *TB_RIGHTArrow;
    QToolButton *TB_BOTTOMArrow;
    QLineEdit *LE_TransValue;
    QToolButton *TB_XRotation;
    QToolButton *TB_YRotation;
    QToolButton *TB_ZRotation;
    QLineEdit *LE_AngleValue;

    void setupUi(QDialog *DLG_ControlBody)
    {
        if (DLG_ControlBody->objectName().isEmpty())
            DLG_ControlBody->setObjectName(QStringLiteral("DLG_ControlBody"));
        DLG_ControlBody->resize(217, 118);
        TB_UPArrow = new QToolButton(DLG_ControlBody);
        TB_UPArrow->setObjectName(QStringLiteral("TB_UPArrow"));
        TB_UPArrow->setGeometry(QRect(39, 10, 31, 31));
        QIcon icon;
        icon.addFile(QStringLiteral("Resources/ic_upArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_UPArrow->setIcon(icon);
        TB_LEFTArrow = new QToolButton(DLG_ControlBody);
        TB_LEFTArrow->setObjectName(QStringLiteral("TB_LEFTArrow"));
        TB_LEFTArrow->setGeometry(QRect(10, 40, 31, 31));
        QIcon icon1;
        icon1.addFile(QStringLiteral("Resources/ic_leftArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_LEFTArrow->setIcon(icon1);
        TB_RIGHTArrow = new QToolButton(DLG_ControlBody);
        TB_RIGHTArrow->setObjectName(QStringLiteral("TB_RIGHTArrow"));
        TB_RIGHTArrow->setGeometry(QRect(69, 40, 31, 31));
        QIcon icon2;
        icon2.addFile(QStringLiteral("Resources/ic_rightArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_RIGHTArrow->setIcon(icon2);
        TB_BOTTOMArrow = new QToolButton(DLG_ControlBody);
        TB_BOTTOMArrow->setObjectName(QStringLiteral("TB_BOTTOMArrow"));
        TB_BOTTOMArrow->setGeometry(QRect(39, 69, 31, 31));
        QIcon icon3;
        icon3.addFile(QStringLiteral("Resources/ic_bottomArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_BOTTOMArrow->setIcon(icon3);
        LE_TransValue = new QLineEdit(DLG_ControlBody);
        LE_TransValue->setObjectName(QStringLiteral("LE_TransValue"));
        LE_TransValue->setGeometry(QRect(40, 40, 30, 30));
        TB_XRotation = new QToolButton(DLG_ControlBody);
        TB_XRotation->setObjectName(QStringLiteral("TB_XRotation"));
        TB_XRotation->setGeometry(QRect(170, 70, 30, 30));
        QIcon icon4;
        icon4.addFile(QStringLiteral("Resources/ic_xRotArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_XRotation->setIcon(icon4);
        TB_YRotation = new QToolButton(DLG_ControlBody);
        TB_YRotation->setObjectName(QStringLiteral("TB_YRotation"));
        TB_YRotation->setGeometry(QRect(120, 20, 30, 30));
        QIcon icon5;
        icon5.addFile(QStringLiteral("Resources/ic_yRotArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_YRotation->setIcon(icon5);
        TB_ZRotation = new QToolButton(DLG_ControlBody);
        TB_ZRotation->setObjectName(QStringLiteral("TB_ZRotation"));
        TB_ZRotation->setGeometry(QRect(120, 70, 30, 30));
        QIcon icon6;
        icon6.addFile(QStringLiteral("Resources/ic_zRotArrow.png"), QSize(), QIcon::Normal, QIcon::Off);
        TB_ZRotation->setIcon(icon6);
        LE_AngleValue = new QLineEdit(DLG_ControlBody);
        LE_AngleValue->setObjectName(QStringLiteral("LE_AngleValue"));
        LE_AngleValue->setGeometry(QRect(170, 20, 30, 30));

        retranslateUi(DLG_ControlBody);

        QMetaObject::connectSlotsByName(DLG_ControlBody);
    } // setupUi

    void retranslateUi(QDialog *DLG_ControlBody)
    {
        DLG_ControlBody->setWindowTitle(QApplication::translate("DLG_ControlBody", "Dialog", Q_NULLPTR));
        TB_UPArrow->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_LEFTArrow->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_RIGHTArrow->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_BOTTOMArrow->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_XRotation->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_YRotation->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
        TB_ZRotation->setText(QApplication::translate("DLG_ControlBody", "...", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_ControlBody: public Ui_DLG_ControlBody {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONTROLBODY_H
