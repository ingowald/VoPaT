#pragma once

#include <memory>
#include <QDialog>
#include "vopat/common.h"
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

