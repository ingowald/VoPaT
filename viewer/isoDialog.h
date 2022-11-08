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

#pragma once

#include <memory>
#include <QDialog>
#include "common/vopat.h"
#include "ui_isoDialog.h"

namespace vopat {

  class IsoDialog : public QDialog
  {
    Q_OBJECT
  public:
    IsoDialog(interval<float> range);

    bool setISOs(const std::vector<int> &active,
                 const std::vector<float> &values,
                 const std::vector<vec3f> &colors);
  signals:
    void isoColorChanged(int iso, QColor clr);
    void isoToggled(int iso, bool enabled);
    void isoValueChanged(int iso, float value);
  private:
    std::unique_ptr<Ui::isoDialog> ui;

    interval<float> valueRange{0.f,1.f};
  };

} // ::vopat

