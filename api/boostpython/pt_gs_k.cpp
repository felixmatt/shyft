
//#include <boost/python.hpp>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/handle.hpp>

#include "core/utctime_utilities.h"

static char const* version() {
   return "v1.0";
}


//typedef std::vector<utcperiod> UtcPeriodVector;
using namespace shyft::core;
using namespace boost::python;
using namespace std;

struct period_handler {
    // this one generates exception to_python(by-value) converter not found
    shared_ptr<calendar const> core_calendar() const {
        return make_shared<calendar const>("Europe/Oslo");
    }
    calendar core_calendar2() {
        return calendar("Europe/Oslo");
    }
	utcperiod period_now() {
		utctime t=utctime_now();
		return utcperiod(t,t+deltahours(1));
	}
	utcperiod trim_period(utcperiod p) {
		calendar utc;
		return utcperiod(utc.trim(p.start,deltahours(1)),utc.trim(p.end,deltahours(1)));
	}
};
//extern void def_api();
BOOST_PYTHON_MODULE(_pt_gs_k)
{

    scope().attr("__doc__")="SHyFT python api for the pt_gs_k model";
    def("version", version);
    class_<period_handler>("PeriodHandler","Just for testing common types")
	.def("period_now",&period_handler::period_now,"testing common type references on return type")
	.def("trim_period",&period_handler::trim_period,args("p"),"trim period to hour")
	.def("calendar",&period_handler::core_calendar,"a calendar, but is this a real calendar ?")
	.def("calendar2",&period_handler::core_calendar2,"a calendar, but is this a real calendar ?");
	//register_ptr_to_python<shared_ptr<const calendar> >();
	//def_api();// pull in common type converters in the api
}
