#include "boostpython_pch.h"
#include <boost/python/docstring_options.hpp>

char const* version() {
   return "v1.1";
}
namespace expose {
    extern void api_geo_point();
    extern void api_geo_cell_data();
    extern void calendar_and_time();
    extern void vectors();
    extern void api_time_axis();
    extern void timeseries();
    extern void target_specification();
    extern void region_environment() ;
    extern void priestley_taylor();
    extern void actual_evapotranspiration();
    extern void gamma_snow();
    extern void kirchner();
    extern void precipitation_correction();
    extern void hbv_snow();
    extern void cell_environment();
    extern void interpolation();
    extern void skaugen_snow();
    extern void kalman();
	extern void hbv_soil();
	extern void hbv_tank();
	extern void hbv_actual_evapotranspiration();
	extern void glacier_melt();
	extern void routing();
    extern void api_cell_state_id();
    
    static std::vector<char> byte_vector_from_file(std::string path) {
        using namespace std;
        ostringstream buf;
        ifstream input;
        input.open(path.c_str(),ios::in|ios::binary);
        if (input.is_open()) {
            buf << input.rdbuf();
            auto s = buf.str();
            return std::vector<char>(begin(s), end(s));
        } else
            throw runtime_error(string("failed to open file for read:") + path);
    }

    static void byte_vector_to_file(std::string path, const std::vector<char>&bytes) {
        using namespace std;
        ofstream out;
        out.open(path, ios::out | ios::binary | ios::trunc);
        if (out.is_open()) {
            out.write(bytes.data(), bytes.size());
            out.flush();
            out.close();
        } else {
            throw runtime_error(string("failed to open file for write:") + path);
        }
    }

    void api() {
        calendar_and_time();
        vectors();
        api_time_axis();
        api_geo_point();
        api_geo_cell_data();
        timeseries();
        target_specification();
        region_environment();
        precipitation_correction();
        priestley_taylor();
        actual_evapotranspiration();
        gamma_snow();
        skaugen_snow();
        hbv_snow();
        kirchner();
        cell_environment();
        interpolation();
        kalman();
		hbv_soil();
		hbv_tank();
		hbv_actual_evapotranspiration();
		glacier_melt();
		routing();
        api_cell_state_id();
        using namespace boost::python;
        def("byte_vector_from_file", byte_vector_from_file, (arg("path")), "reads specified file and returns its contents as a ByteVector");
        def("byte_vector_to_file", byte_vector_to_file, (arg("path"), arg("byte_vector")), "write the supplied ByteVector to file as specified by path");
    }
}

BOOST_PYTHON_MODULE(_api)
{
    boost::python::scope().attr("__doc__")="SHyFT python api providing basic types";
    boost::python::def("version", version);
    boost::python::docstring_options doc_options(true,true,false);// all except c++ signatures
    expose::api();
}
