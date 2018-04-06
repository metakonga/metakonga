/********************************************************************************
** Form generated from reading UI file 'newModel.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_NEWMODEL_H
#define UI_NEWMODEL_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_newModelDialog
{
public:
    QGroupBox *GBNewModel;
    QPushButton *PBOK;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *layout_1;
    QLabel *LName;
    QLabel *LUnit;
    QLabel *LGravity;
    QVBoxLayout *Layout_2;
    QLineEdit *LEName;
    QComboBox *CBUnit;
    QComboBox *CBGravity;
    QLabel *LOpenModel;
    QPushButton *PBBrowse;

    void setupUi(QDialog *newModelDialog)
    {
        if (newModelDialog->objectName().isEmpty())
            newModelDialog->setObjectName(QStringLiteral("newModelDialog"));
        newModelDialog->setWindowModality(Qt::ApplicationModal);
        newModelDialog->resize(440, 178);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(newModelDialog->sizePolicy().hasHeightForWidth());
        newModelDialog->setSizePolicy(sizePolicy);
        newModelDialog->setMinimumSize(QSize(440, 178));
        newModelDialog->setMaximumSize(QSize(440, 178));
        newModelDialog->setModal(true);
        GBNewModel = new QGroupBox(newModelDialog);
        GBNewModel->setObjectName(QStringLiteral("GBNewModel"));
        GBNewModel->setGeometry(QRect(10, 10, 421, 131));
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        GBNewModel->setFont(font);
        GBNewModel->setAutoFillBackground(false);
        PBOK = new QPushButton(GBNewModel);
        PBOK->setObjectName(QStringLiteral("PBOK"));
        PBOK->setGeometry(QRect(330, 23, 75, 91));
        QFont font1;
        font1.setPointSize(34);
        PBOK->setFont(font1);
        widget = new QWidget(GBNewModel);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(11, 23, 311, 91));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        layout_1 = new QVBoxLayout();
        layout_1->setObjectName(QStringLiteral("layout_1"));
        LName = new QLabel(widget);
        LName->setObjectName(QStringLiteral("LName"));

        layout_1->addWidget(LName);

        LUnit = new QLabel(widget);
        LUnit->setObjectName(QStringLiteral("LUnit"));

        layout_1->addWidget(LUnit);

        LGravity = new QLabel(widget);
        LGravity->setObjectName(QStringLiteral("LGravity"));

        layout_1->addWidget(LGravity);


        horizontalLayout->addLayout(layout_1);

        Layout_2 = new QVBoxLayout();
        Layout_2->setObjectName(QStringLiteral("Layout_2"));
        LEName = new QLineEdit(widget);
        LEName->setObjectName(QStringLiteral("LEName"));

        Layout_2->addWidget(LEName);

        CBUnit = new QComboBox(widget);
        CBUnit->setObjectName(QStringLiteral("CBUnit"));

        Layout_2->addWidget(CBUnit);

        CBGravity = new QComboBox(widget);
        CBGravity->setObjectName(QStringLiteral("CBGravity"));

        Layout_2->addWidget(CBGravity);


        horizontalLayout->addLayout(Layout_2);

        LOpenModel = new QLabel(newModelDialog);
        LOpenModel->setObjectName(QStringLiteral("LOpenModel"));
        LOpenModel->setGeometry(QRect(20, 150, 81, 16));
        PBBrowse = new QPushButton(newModelDialog);
        PBBrowse->setObjectName(QStringLiteral("PBBrowse"));
        PBBrowse->setGeometry(QRect(340, 147, 75, 23));
        PBBrowse->setStyleSheet(QStringLiteral("border-bottom-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 255), stop:0.339795 rgba(255, 0, 0, 255), stop:0.339799 rgba(255, 255, 255, 255), stop:0.662444 rgba(255, 255, 255, 255), stop:0.662469 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));"));
        PBBrowse->setFlat(false);

        retranslateUi(newModelDialog);

        PBBrowse->setDefault(false);


        QMetaObject::connectSlotsByName(newModelDialog);
    } // setupUi

    void retranslateUi(QDialog *newModelDialog)
    {
        newModelDialog->setWindowTitle(QApplication::translate("newModelDialog", "New Model", Q_NULLPTR));
        GBNewModel->setTitle(QApplication::translate("newModelDialog", "New Model", Q_NULLPTR));
        PBOK->setText(QApplication::translate("newModelDialog", "OK", Q_NULLPTR));
        LName->setText(QApplication::translate("newModelDialog", "Name", Q_NULLPTR));
        LUnit->setText(QApplication::translate("newModelDialog", "Unit", Q_NULLPTR));
        LGravity->setText(QApplication::translate("newModelDialog", "Gravity", Q_NULLPTR));
        CBUnit->clear();
        CBUnit->insertItems(0, QStringList()
         << QApplication::translate("newModelDialog", "MKS(Meter/Kilogram/Newton/Second)", Q_NULLPTR)
        );
        CBGravity->clear();
        CBGravity->insertItems(0, QStringList()
         << QApplication::translate("newModelDialog", "+X", Q_NULLPTR)
         << QApplication::translate("newModelDialog", "+Y", Q_NULLPTR)
         << QApplication::translate("newModelDialog", "+Z", Q_NULLPTR)
         << QApplication::translate("newModelDialog", "-X", Q_NULLPTR)
         << QApplication::translate("newModelDialog", "-Y", Q_NULLPTR)
         << QApplication::translate("newModelDialog", "-Z", Q_NULLPTR)
        );
        LOpenModel->setText(QApplication::translate("newModelDialog", "Open Model", Q_NULLPTR));
        PBBrowse->setText(QApplication::translate("newModelDialog", "Browse", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class newModelDialog: public Ui_newModelDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NEWMODEL_H
