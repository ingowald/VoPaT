#include <cassert>
#include <QColorDialog>
#include "isoDialog.h"

namespace vopat {

  IsoDialog::IsoDialog(interval<float> range)
    : ui(new Ui::isoDialog)
    , valueRange(range)
  {
    ui->setupUi(this);

    auto onEnableClicked = [this](int state) {
      bool enabled = state==Qt::Checked;
      QCheckBox* cb = qobject_cast<QCheckBox*>(QObject::sender());
      if (cb==ui->cbISO1) {
        emit isoToggled(0,enabled);
      } else if (cb==ui->cbISO2) {
        emit isoToggled(1,enabled);
      } else if (cb==ui->cbISO3) {
        emit isoToggled(2,enabled);
      } else if (cb==ui->cbISO4) {
        emit isoToggled(3,enabled);
      } else assert(0);
    };

    connect(ui->cbISO1,&QCheckBox::stateChanged,this,onEnableClicked);
    connect(ui->cbISO2,&QCheckBox::stateChanged,this,onEnableClicked);
    connect(ui->cbISO3,&QCheckBox::stateChanged,this,onEnableClicked);
    connect(ui->cbISO4,&QCheckBox::stateChanged,this,onEnableClicked);

    auto onColorClicked = [this]() {
      QColorDialog dlg;
      dlg.setOption(QColorDialog::DontUseNativeDialog);
      if (dlg.exec() == QColorDialog::Accepted) {
        QPushButton* pb = qobject_cast<QPushButton*>(QObject::sender());
        if (pb) {
          QColor clr = dlg.currentColor();
          pb->setStyleSheet( "* { background-color: "+ clr.name() + " }");
          if (pb==ui->pbColor1) {
            emit isoColorChanged(0,clr);
          } else if (pb==ui->pbColor2) {
            emit isoColorChanged(1,clr);
          } else if (pb==ui->pbColor3) {
            emit isoColorChanged(2,clr);
          } else if (pb==ui->pbColor4) {
            emit isoColorChanged(3,clr);
          } else assert(0);
        }
      }
    };

    connect(ui->pbColor1,&QPushButton::clicked,this,onColorClicked);
    connect(ui->pbColor2,&QPushButton::clicked,this,onColorClicked);
    connect(ui->pbColor3,&QPushButton::clicked,this,onColorClicked);
    connect(ui->pbColor4,&QPushButton::clicked,this,onColorClicked);

    auto onValueChanged = [this](int value) {
      QSlider* slider = qobject_cast<QSlider*>(QObject::sender());
      if (!slider)
        return;
      float val01 = (value-slider->minimum())/((float)slider->maximum()-slider->minimum());
      float valf = (1.f-val01) * valueRange.lo + val01 * valueRange.hi;
      if (slider==ui->sliderISO1) {
        emit isoValueChanged(0,valf);
      } else if (slider==ui->sliderISO2) {
        emit isoValueChanged(1,valf);
      } else if (slider==ui->sliderISO3) {
        emit isoValueChanged(2,valf);
      } else if (slider==ui->sliderISO4) {
        emit isoValueChanged(3,valf);
      } else assert(0);
    };

    connect(ui->sliderISO1,&QSlider::valueChanged,this,onValueChanged);
    connect(ui->sliderISO2,&QSlider::valueChanged,this,onValueChanged);
    connect(ui->sliderISO3,&QSlider::valueChanged,this,onValueChanged);
    connect(ui->sliderISO4,&QSlider::valueChanged,this,onValueChanged);
  }

} // ::vopat

