// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <cassert>
#include <QColorDialog>
#include "isoDialog.h"
#include "model/ModelConfig.h"

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

  bool IsoDialog::setISOs(const std::vector<int> &active,
                          const std::vector<float> &values,
                          const std::vector<vec3f> &colors)
  {
    if (active.size() != values.size() ||
        active.size() != colors.size() ||
        active.size() > ModelConfig::maxISOs)
      return false;

    std::vector<QCheckBox *> checkBoxes(
      {ui->cbISO1,ui->cbISO2,ui->cbISO3,ui->cbISO4});

    std::vector<QPushButton *> colorButtons(
      {ui->pbColor1,ui->pbColor2,ui->pbColor3,ui->pbColor4});

    std::vector<QSlider *> sliders(
      {ui->sliderISO1,ui->sliderISO2,ui->sliderISO3,ui->sliderISO4});

    for (size_t i=0; i<active.size(); ++i) {
      checkBoxes[i]->setCheckState(active[i]?Qt::Checked:Qt::Unchecked);
      QColor clr({int(colors[i].x*255),int(colors[i].y*255),int(colors[i].z*255)});
      colorButtons[i]->setStyleSheet( "* { background-color: "+ clr.name() + " }");
      float val01 = (values[i]-valueRange.lo)/(valueRange.hi-valueRange.lo);
      sliders[i]->setValue(int(1.f-val01) * sliders[i]->minimum() + val01 * sliders[i]->maximum());

      emit isoToggled((int)i,active[i]?true: false);
      emit isoColorChanged((int)i,clr);
      emit isoValueChanged((int)i,values[i]);
    }

    return true;
  }

} // ::vopat

