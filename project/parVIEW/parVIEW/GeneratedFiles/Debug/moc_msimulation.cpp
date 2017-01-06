/****************************************************************************
** Meta object code from reading C++ file 'msimulation.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.4.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../common/inc/msimulation.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'msimulation.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.4.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_simulation_t {
    QByteArrayData data[6];
    char stringdata[48];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_simulation_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_simulation_t qt_meta_stringdata_simulation = {
    {
QT_MOC_LITERAL(0, 0, 10), // "simulation"
QT_MOC_LITERAL(1, 11, 8), // "finished"
QT_MOC_LITERAL(2, 20, 0), // ""
QT_MOC_LITERAL(3, 21, 12), // "sendProgress"
QT_MOC_LITERAL(4, 34, 6), // "cpuRun"
QT_MOC_LITERAL(5, 41, 6) // "gpuRun"

    },
    "simulation\0finished\0\0sendProgress\0"
    "cpuRun\0gpuRun"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_simulation[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x06 /* Public */,
       3,    1,   35,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    0,   38,    2, 0x0a /* Public */,
       5,    0,   39,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::UInt,    2,

 // slots: parameters
    QMetaType::Bool,
    QMetaType::Bool,

       0        // eod
};

void simulation::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        simulation *_t = static_cast<simulation *>(_o);
        switch (_id) {
        case 0: _t->finished(); break;
        case 1: _t->sendProgress((*reinterpret_cast< uint(*)>(_a[1]))); break;
        case 2: { bool _r = _t->cpuRun();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 3: { bool _r = _t->gpuRun();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (simulation::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&simulation::finished)) {
                *result = 0;
            }
        }
        {
            typedef void (simulation::*_t)(unsigned int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&simulation::sendProgress)) {
                *result = 1;
            }
        }
    }
}

const QMetaObject simulation::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_simulation.data,
      qt_meta_data_simulation,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *simulation::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *simulation::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_simulation.stringdata))
        return static_cast<void*>(const_cast< simulation*>(this));
    return QObject::qt_metacast(_clname);
}

int simulation::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void simulation::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void simulation::sendProgress(unsigned int _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
