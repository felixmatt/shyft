#include "boostpython_pch.h"

#include "py_convertible.h"

#include "core/utctime_utilities.h"
#include "core/time_axis.h"
#include "core/timeseries.h"
#include "core/cell_model.h"

namespace expose {
    using namespace boost::python;

    void cell_environment() {

        typedef shyft::core::environment_t CellEnvironment;
        class_<CellEnvironment>("CellEnvironment","Contains all ts projected to a certain cell-model using interpolation step (if needed)")
            .def_readwrite("temperature",&CellEnvironment::temperature)
            .def_readwrite("precipitation",&CellEnvironment::precipitation)
            .def_readwrite("radiation",&CellEnvironment::radiation)
            .def_readwrite("wind_speed",&CellEnvironment::wind_speed)
            .def_readwrite("rel_hum",&CellEnvironment::rel_hum)
            .def("init",&CellEnvironment::init,args("ta"),"zero all series, set time-axis ta")
            ;

        typedef shyft::core::environment_const_rhum_and_wind_t CellEnvironmentConstRHumWind;
        class_<CellEnvironmentConstRHumWind>("CellEnvironmentConstRHumWind","Contains all ts projected to a certain cell-model using interpolation step (if needed)")
            .def_readwrite("temperature",&CellEnvironmentConstRHumWind::temperature)
            .def_readwrite("precipitation",&CellEnvironmentConstRHumWind::precipitation)
            .def_readwrite("radiation",&CellEnvironmentConstRHumWind::radiation)
            .def_readwrite("wind_speed",&CellEnvironmentConstRHumWind::wind_speed)
            .def_readwrite("rel_hum",&CellEnvironmentConstRHumWind::rel_hum)
            .def("init",&CellEnvironmentConstRHumWind::init,args("ta"),"zero all series, set time-axis ta")
            ;

    }
}
