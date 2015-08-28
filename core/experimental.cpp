#include "core_pch.h"
#include "experimental.h"

// for test purposes, we need relative addressing of file paths, easiest to use boost
#include <boost/filesystem.hpp>

namespace shyft {
    namespace experimental {
        namespace io {
            /**\brief in a clever way, figure out the root directory for tests so that we can use relative paths for the remaining names */
            static boost::filesystem::path test_root_dir() {
                auto cwd=boost::filesystem::current_path();
                boost::filesystem::path test_path;
                test_path= cwd/".."/".."/"test"; // assume current dir is <enkiroot>/bin/Debug|Release
                test_path.normalize();
                return test_path;
            }
            /**\brief given a relative path, compute the working (abs) path to the file, print out error if not exists */
            string test_path(string rel_path) {
                auto pth=  test_root_dir();
                pth.append(rel_path);
                pth.normalize();
                if(!is_regular_file(pth)) {
                    cout<<"Error: missing:"<<pth<<", cwd is:"<<test_root_dir()<<endl;
                }
                string  r=pth.string();
                return r;
            }
            ///< 'slurp' a file into a  memory string string
            std::string slurp (const std::string& path) {
                std::ostringstream buf;
                std::ifstream input (path.c_str());
                buf << input.rdbuf();
                return buf.str();
            }
            /** \brief  given a subdir, search for all files with matching suffix in that directory, return back list of paths */
            vector<string> find(const string subdir,const string &suffix) {
                auto root=test_root_dir();
                using namespace boost::filesystem;
                root.append(subdir);
                root.normalize();
                vector<string> r;
                if(is_directory(root)) {
                    for(auto i =directory_iterator(root); i!= directory_iterator();++i) {
                        if(is_regular_file(*i) ) {
                            string fn=(*i).path().filename().string();
                            if(fn.size()>suffix.size() && fn.find(suffix)==fn.size()-suffix.size())
                                r.push_back(fn);
                        }
                    }
                }
                return r;
            }
            /** given a wkt_reader (to help convert fro string to geo_xts_t), a lambda to convert from geo_id to geo_point, a subdir and a suffix,
             * read all files matching and provide them back as a shared pointer to a vector of \ref geo_xts_t */
            shared_ptr<vector<geo_xts_t>>
            load_from_directory(wkt_reader& wkt_io,function<ec::geo_point(int)> id_to_geo_point,const string& subdir,const string& suffix) {
                auto filenames=find(subdir,suffix);
                auto r= make_shared<vector<geo_xts_t>>();
                for(auto f:filenames) {
                    //cout<<"slurp:"<<f<<endl;
                    r->push_back(wkt_io.read_geo_xts_t(suffix+":"+f,id_to_geo_point,slurp(test_path(subdir+"/"+f))));
                }
                return r;
            }

        }
        namespace repository {
        }
    } // experimental
} // shyft
