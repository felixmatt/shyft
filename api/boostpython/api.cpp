#include "boostpython_pch.h"
#include <fstream>

char const* version() {
   return "v4.x";
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
    extern void universal_snow();
    extern void kirchner();
    extern void precipitation_correction();
    extern void hbv_snow();
    extern void hbv_physical_snow();
    extern void cell_environment();
    extern void interpolation();
    extern void skaugen_snow();
    extern void kalman();
	extern void hbv_soil();
	extern void hbv_tank();
	extern void hbv_actual_evapotranspiration();
	extern void glacier_melt();
	extern void routing();
	extern void dtss();
    extern void dtss_finalize();
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
    static std::vector<char> byte_vector_from_hex_str(const std::string& s) {
        using namespace std;
        vector<char> r;
        if(s.size()==0)
            return r;
        if(s.size() %2 )
            throw runtime_error(string("hex_str should be even-sized"));
        r.reserve(s.size()/2);
        for(size_t i=0;i<s.size();i+=2) {
            uint32_t b=0;
            if(sscanf(s.c_str()+i,"%02x",&b)==1) {
                r.push_back(b&0x00ffu);
            } else {
                throw runtime_error(string("illegal hex at ")+to_string(i)+ string(" offset "));
            }
        }
        return r;
    }
    static std::string byte_vector_to_hex_str(const std::vector<char>&b) {
        using namespace std;
        string r;
        r.reserve(b.size()*2);
        for(const unsigned char c:b){
            char s[10];
            sprintf(s,"%02x",c); // c
            r.push_back(s[0]);
            r.push_back(s[1]);
        }
        return r;
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
        hbv_physical_snow();
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
        def("byte_vector_from_hex_str", byte_vector_from_hex_str, (arg("hex_str")), "converts earlier byte-vector hex-string to byte-vector");
        def("byte_vector_to_hex_str", byte_vector_to_hex_str, (arg("byte_vector")), "return hex-string of byte-vector");
        dtss();
    }
}
void finalize_api() {
    //extern void expose::dtss_finalize();
    expose::dtss_finalize();
}
#ifdef _WIN32
#include "Windows.h"
std::string win_short_path(const std::string& long_path) {
    long length = GetShortPathName(long_path.c_str(), NULL, 0);
    std::string r(length+1, '\0');
    length = GetShortPathName(long_path.c_str(),(char*) r.data(), length);
    r.resize(length);
    return r;
}
#else
std::string win_short_path(const std::string& long_path) {
    return long_path;
}
#endif

BOOST_PYTHON_MODULE(_api) {
    namespace py = boost::python;
    py::scope().attr("__doc__") = "SHyFT python api providing basic types";
    py::def("version", version);
    py::docstring_options doc_options(true, true, false);// all except c++ signatures
    expose::api();
    // We register the function with the atexit module which
    // will be called _before_ the Python C-API is unloaded.
    // needed for proper clean-up on windows platform
    // otherwise python hangs on dlib::shared_ptr_thread_safe destruction
    py::def("win_short_path", win_short_path, py::arg("path"),
            doc_intro("WinApi function GetShortPath exposed to python")
            doc_intro("https://msdn.microsoft.com/en-us/library/windows/desktop/aa364989(v=vs.85).aspx")
            doc_intro("Note that it only works for file-paths that exists, returns null string for not-existing files")
            doc_parameters()
            doc_parameter("path","str","a long path form")
            doc_returns("short_path","str","windows 8.3 path string if on windows for *existing* files, otherwise same as input path")
            );
    py::def("_finalize", &finalize_api);
    py::object atexit = py::object(py::handle<>(PyImport_ImportModule("atexit")));
    py::object finalize = py::scope().attr("_finalize");
    atexit.attr("register")(finalize);
}

