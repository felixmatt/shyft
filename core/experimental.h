#pragma once
#ifdef SHYFT_NO_PCH
#define BOOST_GEOMETRY_OVERLAY_NO_THROW

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>
#include <boost/geometry/index/rtree.hpp>

#endif // SHYFT_NO_PCH
#include "geo_cell_data.h"
#include "utctime_utilities.h"
#include "time_series.h"

#include "region_model.h"
#include "pt_gs_k_cell_model.h"



namespace shyft {
    using  namespace std;
    /** \brief experimental classes to support testing in C++ */
    namespace experimental {
        namespace geo = boost::geometry;///< we are using boost::geometry to take care of everything that needs geo computation
        namespace ec = shyft::core;
        namespace et = shyft::time_series;
		namespace ta = shyft::time_axis;
        /* map geo types into this namespace,*/
        typedef geo::model::d2::point_xy<double> point_xy;
        typedef geo::model::polygon<point_xy> polygon;
        typedef geo::model::multi_polygon<polygon> multi_polygon;
        typedef geo::model::multi_point<point_xy> multi_point;
        typedef geo::model::box<point_xy> box;

        /** \note using trailing underscore (_) convention for shared_ptr version */
        typedef shared_ptr<multi_polygon> multi_polygon_;
        typedef shared_ptr<multi_point> multi_point_;

        /// and typedefs for commonly used types in this test
        typedef et::point_ts<ta::fixed_dt>      pts_t;
        //typedef et::point_source_with_no_regularity xts_t;
        typedef et::point_ts<ta::point_dt> xts_t;
        typedef et::constant_timeseries<ta::fixed_dt>   cts_t;


        /// geolocated versions of regular fixed interval ts, irregular interval ts and constant ts
        typedef ec::geo_point_ts<pts_t> geo_pts_t;
        typedef ec::geo_point_ts<xts_t> geo_xts_t;
        typedef ec::geo_point_ts<cts_t> geo_cts_t;

        /**\brief construct the region_environment we need, currently we have only constant humidity and windspeed so these are 'constant' ts type */
        typedef ec::region_environment<geo_xts_t,geo_xts_t,geo_xts_t,geo_cts_t,geo_cts_t> region_environment_t;

        /** \brief simple dtm representation, a box and a z., later it could be + aspect,slope etc.(ref wiki geo aspect/slope)
         */
        struct dtm {
            box bbox;///< the bounding box of the grid
            double z;///< represents the average elevation in meter, over the average of the bbox
        };
        /** \brief simplest possible observation_location */
        struct observation_location {
            string name;//<common name of the location
            ec::geo_point point;//< shyft::core::geo_point specifies the full xyz of the named location
        };

        /** \brief wkt_reader, WellKnownTextformat(?) reader:
         *   Just for the testing and experimental purposes,we need something that gives a reasonable amount of data
         *  that is realistic.
         *  From Statkraft we have the nea-nidelv system, and this reader is used to parse/read those files
         * into memory/structures that we can use to construct cells, environmental time-series, etc. that are
         * needed to run a full model.
         *   (and GIS system might provide 'non-perfect' data that we need to handle)
         * \note this class contains readers for the files located in test/neanidelv directory
         */
        class wkt_reader {
        public:
            vector<string> reported_errors;///< contains accumulated errors after calls to read_xxx
            /**\brief internally used to collect errors, using report_tag to easy locate errors in files|file-types */
            void report_error(string report_tag,int line_no,string msg) {
                ostringstream os;
                os<<report_tag<<"("<<line_no<<"):"<<msg;
                reported_errors.emplace_back(os.str());
            }
            /** \brief given string s containing txt formatted as dtm(box(point(),point()),z) , convert it to a vector<dtm> \ref dtm */
            vector<dtm> read_dtm(string report_tag,string s) {
                istringstream cs(s);
                string line;
                int line_no=0;
                vector<dtm> r;
                const char *dtm_format="dtm(box(point(%lf,%lf),point(%lf,%lf)),%lf)";
                while( getline(cs,line)) {
                    ++line_no;
                    double x0,y0,x1,y1,z;
                    if(sscanf(line.c_str(),dtm_format,&x0,&y0,&x1,&y1,&z)==5) {
                        dtm x{box(point_xy(x0,y0),point_xy(x1,y1)),z};
                        r.emplace_back(x);
                    } else {
                        report_error(report_tag,line_no,string("Invalid dtm spec,expects")+ string(dtm_format));
                    }
                }
                return r;
            }
            /** \brief Given string s with wkt formatted multi_polygons, convert the lines into a multi_polygon. \note supports ! as comment in the format */
            multi_polygon_ read(string report_tag,string s ) {
                istringstream cs(s);
                string line;
                int line_no=0;
                auto r=make_shared<multi_polygon>();
                while( getline(cs,line)) {
                    ++line_no;
                    if(line.size()&& line[0]=='!')
                        continue;//skip comments
                    size_t eol=  line.find_last_of(')');
                    if(eol !=string::npos) {
                        polygon p;
                        try {
                            geo::read_wkt(line.substr(0,eol+1),p);
                            if(!geo::is_valid(p)) {
                                geo::correct(p);
                            }
                            if(geo::is_valid(p)){
                                r->push_back(p);
                            } else {
                                report_error(report_tag,line_no,"Invalid and not repairable polygon, please inspect the file");
                            }
                        } catch(const geo::read_wkt_exception&ex) {
                            report_error(report_tag,line_no,ex.what());
                        }
                    }
                }
                return r;
            }

            /**\brief given string s formatted with wkt point structures, read from the string into a multipoint */
            multi_point_ read_points(string report_tag ,string s) {
                istringstream cs(s);
                string line;
                int line_no=0;
                auto r=make_shared<multi_point>();
                while( getline(cs,line)) {
                    ++line_no;
                    size_t eol=  line.find_last_of(')');
                    if(eol !=string::npos) {
                        point_xy p;
                        try {
                            geo::read_wkt(line.substr(0,eol+1),p);
                            r->push_back(p);
                        } catch(const geo::read_wkt_exception&ex) {
                            report_error(report_tag,line_no,ex.what());
                        }
                    }
                }
                return r;
            }

            /**\brief given string formatted as id:wkt multipolygon, read and convert the string to a map<id,multi_polygon_> */
            map<int,multi_polygon_> read_catchment_map(string report_tag,string s) {
                istringstream cs(s);
                string line;
                int line_no=0;
                map<int,multi_polygon_> r;
                while( getline(cs,line)) {
                    ++line_no;
                    size_t split=line.find_first_of(':');
                    size_t eol=  line.find_last_of(')');
                    if(split != string::npos && eol !=string::npos) {
                        int cid=stoi(line.substr(0,split));
                        polygon p;
                        try {
                            geo::read_wkt(line.substr(split+1,eol-split),p);
                            if(!geo::is_valid(p)) {
                                geo::correct(p);
                            }
                            if(geo::is_valid(p)){
                                if(r.find(cid)==end(r)) {
                                    auto mp=make_shared<multi_polygon>();mp->push_back(p);
                                    r.insert(pair<int,multi_polygon_>(cid,mp));
                                } else {
                                    r[cid]->push_back(p);
                                }
                            } else {
                                report_error(report_tag,line_no,"Sorry, got bad poly in file");
                            }
                        } catch(const geo::read_wkt_exception&ex) {
                            report_error(report_tag,line_no,ex.what());
                        }
                    }
                }
                return r;
            }
            /**\brief given string s formatted as id:name:POINT(x,y,z), convert it to a map<id,observation_location>
            * used to ensure that locations (x,y,z) associated with observation time-series have consistent location info */
            map<int,observation_location> read_geo_point_map(string report_tag,string s) {
                map<int,observation_location> r;
                istringstream cs(s);
                string line;
                int line_no=0;
                const char *lformat="%d:%*[^:]:POINT(%lf,%lf,%lf)";
                while( getline(cs,line)) {
                    ++line_no;
                    int id;
                    double x,y,z;

                    if(sscanf(line.c_str(),lformat,&id,&x,&y,&z)==4) {
                        observation_location l;
                        auto f =line.find_first_of(':');
                        auto e =line.find_last_of(':');
                        l.name=line.substr(f,e-f);
                        l.point.x=x;
                        l.point.y=y;
                        l.point.z=z;
                        r.insert(make_pair(id,l));
                    } else {
                        report_error(report_tag,line_no,string("Invalid obs loc spec,expects")+ string(lformat));
                    }
                }
                return r;

            }
            /**\brief given string formatted as 'geo-id-located ts', read it and convert it to a geo_xts_t
            * used to read time-series for nea-nidelv and similar cases */
            geo_xts_t read_geo_xts_t(string report_tag, function<ec::geo_point(int)> to_geo_point, const string& s) {
                istringstream cs(s);
                string line;
                ec::calendar cal;
                int line_no=0;
                geo_xts_t r;
                vector<ec::utctime> time_points;
                vector<double> values;
                //r.ts.reserve(s.size()/(25));//approx. sz.pr point
                time_points.reserve(s.size()/(25));
                values.reserve(s.size()/(25));
                while( getline(cs,line)) {
                    line_no++;
                    switch(line_no) {
                        case 1:{
                            int geo_id=0;
                            if(sscanf(line.c_str(),"geo_point_id %d",&geo_id)!=1) {
                                report_error(report_tag,line_no,"expected 1.line staring with geo_point_id <geo_point_id>");
                                return r;
                            }
                            r.location=to_geo_point(geo_id);
                        } break;
                        case 2: {
                            char ts_type[200];ts_type[0]=0;
                            if(sscanf(line.c_str(),"time %40s",ts_type)!=1) {
                                report_error(report_tag,line_no,"expected 2.line staring with time <ts_type>");
                                return r;
                            }
                        } break;
                        default: {
                            int Y,M,D,h,m,s;
                            double v;
                            if(line.find_first_not_of(" \t\n")==string::npos)
                                continue;
                            if(sscanf(line.c_str(),"%04d.%02d.%02d %02d:%02d:%02d %lf",&Y,&M,&D,&h,&m,&s,&v)==7) {
                                //r.ts.add_point(et::point(cal.time(ec::YMDhms(Y,M,D,h,m,s)),v));
                                time_points.emplace_back(cal.time(ec::YMDhms(Y,M,D,h,m,s)));
                                values.emplace_back(v);
                            } else {
                                report_error(report_tag,line_no,"Wrong format, expected YYYY.MM.DD hh:mm:ss value");
                                return r;
                            }
                        } break;
                    }
                }
                time_points.emplace_back(ec::max_utctime);// open-ended time-axis

                shyft::time_axis::point_dt pts(time_points);
                r.ts= xts_t(pts,values,et::POINT_INSTANT_VALUE);
                return r;
            }
        };

        /** \brief A region_grid model, provides a dmz, boundingboxes with a z.
         * The region_grid is used by the geo_cell_data computer to provide the cell-geometry for a grid, along with the
         * elevation information.
         *
         *  TODO: This class is just to provide barely enough functionality to work with the existing test data.
         *
         *  -#: re-work  the class to provide iterator of dtm's instead of (i,j) stuff.
         *  -#: fix the set_dtm to do a reasonable attempt to do projection (using an external lib..)
         *
         */
        class region_grid {
          public:
            region_grid(point_xy p0,size_t nx,size_t ny,double dx=1000,double dy=1000): p0(p0),nx(nx),ny(ny),dx(dx),dy(dy){}
            int set_dtm(const vector<dtm>& dtminfo) {
                //TODO: have to recompute dtminfo into the cell_box boundaries
                dtmz.clear();
                dtmz.reserve(n_cells());
                int misses=0;
                for(size_t i=0;i<nx;++i) {
                    for(size_t j=0;j<ny;++j) {
                        size_t ix=i*ny+j;
                        auto mp(cell_midpoint(i,j));
                        dtmz.push_back(0.0);
                        if(ix<dtminfo.size()) {
                            if(geo::covered_by(mp,dtminfo[ix].bbox)) {
                                dtmz[ix]=dtminfo[ix].z;
                            } else { // linear (in-effecient) search. using the test-data, we should never end here..
                                misses++;
                                for(auto d:dtminfo){
                                    if(geo::covered_by(mp,d.bbox)) {
                                        dtmz[ix]=d.z;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                return misses;
            }
            /**\brief return the bounding box of a cell(i,j) */
            box cell_box(size_t i,size_t j) const {
                return box(point_xy(p0.x()+i*dx,p0.y()+j*dy),point_xy(p0.x()+(i+1)*dx,p0.y()+(j+1)*dy));
            }
            /**\brief calculate and return the midpoint of the cell(i,j) */
            point_xy cell_midpoint(size_t i,size_t j) {
                point_xy mp;geo::centroid(cell_box(i,j),mp);
                return mp;
            }
            /**\brief return the digital terrain model z coordinate for cell(i,j)
             *\returns dtm z elevation, or 0.0 if the supplied (i,j) are outside bound*/
            double dtm_z(size_t i,size_t j) const {
                size_t ix=i*ny+j;
                if(ix<dtmz.size())
                    return dtmz[ix];
                return 0.0;
            }
            /** \brief returns number of cells in the x direction */
            size_t n_x() const {return nx;}
            /** \brief returns number of cells in the y direction */
            size_t n_y() const {return ny;}
            /** \brief returns total number of cells, n_x()*n_y() */
            size_t n_cells() const {return nx*ny;}
          private:
            point_xy p0;///< 2d geo point of "upper left corner"
            size_t nx; ///< number of cells in x direction
            size_t ny; ///< number of cells in y direction
            double dx;///< distance in meters for the x-side of the cell
            double dy;///< distance in meters for the y-side of the cell
            vector<double>  dtmz; ///<< vector containing all the elevations for the cells, computed index cell(i,j) is i*ny+j
        };

        /** \brief A computer that given region_grid, and catchment + features, delivers geo_cell_data filled in back.
         * \note that this class uses the \ref region_grid class to provide the region grid geometry along with the elevation(z).
         */
        class geo_cell_data_computer {
          public:
            geo_cell_data_computer():area_errors(0),suppressed_exceptions(0) {
            }

            vector<ec::geo_cell_data>
            catchment_geo_cell_data(const region_grid& grid,
                                    int catchm_id,
                                    double radiation_factor,
                                    const multi_polygon& catchm,
                                    const multi_polygon& rsvx,
                                    const multi_polygon& lakex,
                                    const multi_polygon& glacierx,
                                    const multi_polygon& forestx
                                    ) {
                vector<ec::geo_cell_data> r;
                r.reserve(grid.n_cells());
                box bbox;
                geo::envelope(catchm,bbox);
                polygon pgbbox;
                geo::convert(bbox,pgbbox);
                // clip features to catchm. first, hoping that limits the exec time.to
                multi_polygon rsv;
                multi_polygon lake;
                multi_polygon glacier;
                multi_polygon forest;

                if(rsvx.size()) geo::intersection(pgbbox,rsvx,rsv);
                if(lakex.size()) geo::intersection(pgbbox,lakex,lake);
                if(glacierx.size()) geo::intersection(pgbbox,glacierx,glacier);
                if(forestx.size()) geo::intersection(pgbbox,forestx,forest);
                //TODO: consider using multicore (not yet successful speedup)
                // using a lambda, e.g. prepared for multicore
                // a mutex for the result
                // uncomment the next lines to prepare for multicore run on the cells.
                //mutex r_mx;
                //r.resize(grid.n_cells());
                auto f_x=[&](size_t lower_i,size_t upper_i,vector<ec::geo_cell_data>&r ) {
                    for(size_t i=lower_i;i<upper_i;++i) {
                        for(size_t j=0;j<grid.n_y();++j) {
                            box cell_bx(grid.cell_box(i,j));
                            if(geo::intersects(cell_bx,bbox)) {
                                multi_polygon cellx;
                                geo::convert(cell_bx,cellx);
                                try {
                                    multi_polygon cell;
                                    geo::intersection(cellx,catchm,cell);//cell_box throws exception, so avoid it.
                                    if(cell.size() ) { // get rid of empty intersections
                                        double a=geo::area(cell);
                                        point_xy midpoint(0,0);
                                        geo::centroid(cell,midpoint);//Could use cell_bx midpoint instead, ..faster
                                        ec::geo_point pc(midpoint.x(),midpoint.y(),grid.dtm_z(i,j));
                                        ec::land_type_fractions ltf(compute_land_type_fractions(a,cell,rsv,lake,glacier,forest));
                                        ec::geo_cell_data gcd(pc,a,catchm_id,radiation_factor,ltf);
                                        {
                                            //lock_guard<mutex> lock(r_mx);
                                            r.emplace_back(gcd);

                                        }
                                    }
                                } catch(const exception &ex) { //geo::overlay_invalid_input_exception, the lake from neanidelv generates that.
                                    cout<<"Sorry, at cell("<<i<<","<<j<<"), intersection ex: "<<ex.what()<<endl;
                                    ++suppressed_exceptions;
                                }
                            }
                        }
                    }
                };
                f_x(0,grid.n_x(),r);// invoke lambda for all cells
                return r;//moved anyway?
            }
            // for debug/diagnostics ,needed when working with real data, (sorry, a GIS db may contain errors!)
            mutable int area_errors;//< count number of errors during the .safe_area_of(..) function
            mutable int suppressed_exceptions;//< count number of exceptions during boost::geo intersect etc..
          private:

            /** \brief calculates the area of the intersection between cell and a feature (like lake/forest),
             *  catching any exceptions
             *
             * \param cell the multi_polygon representing a cell geometry
             * \param feature the multi_polygon representing the feature, like forest, lake,glacier
             * \return area of the intersection between cell and feature, 0.0 if any exception occur
             *
             */
            double safe_area_of(const multi_polygon& cell,const multi_polygon &feature) const {
                try {
                    multi_polygon common;
                    if(cell.size() && feature.size()) {
                        geo::intersection(cell,feature,common);
                        return common.size()?geo::area(common):0.0;
                    } else
                        return 0.0;
                } catch( const std::exception&) {
                    area_errors++;
                    return 0.0;
                }
            }
            /** \brief Given geometry of cell, plus features geometry of reservoir, lake, glacier and forest,
             * calculate the fractions of each feature in a robust manner.
             * Note that the features rsv/lake/glacier and forrest are mutually exclusive within a cell (but this is not the case at the output from the GIS system).
             * Typical problems is that the features are not exclusive, and we even have to deal with
             * some special errors where the rsv_m2 and lake_m2 are equal, then set lake to 0.0 (since we assume that its a reservoir, based on manual check of data).
             * Other issues is that the areas does not add up to 1.0, so we have to calculate the fractions in a robust manner.
             * \return shyft::core::land_type_fractions
             *
             */

            ec::land_type_fractions compute_land_type_fractions( double cell_area,
                                    const multi_polygon& cell ,
                                    const multi_polygon& rsv,
                                    const multi_polygon& lake,
                                    const multi_polygon& glacier,
                                    const multi_polygon& forest ) const {

                double f=safe_area_of(cell,forest);
                double l=safe_area_of(cell,lake);
                double r=safe_area_of(cell,rsv);
                double g=safe_area_of(cell,glacier);

                // but the gis-system is not.. so we have to fix it here
                // rules: if r==l, within 1000 m2 then drop out l (its a reservoir)
                // then use s in stead of a if s > a
                if(fabs(r-l)<1000) l=0.0;
                double s=g+r+l+f;// sum, and they are to be considered exclusive,non-overlapping
                if(s>cell_area) cell_area=s;
                ec::land_type_fractions ltf;
                ltf.set_fractions(g/cell_area,l/cell_area,r/cell_area,f/cell_area);
                return ltf;
            }
        };


        /**\brief io utilities for the experimental stuff to support testing and experiments in c++ mode of work*/
        namespace io {

            string test_path(string rel_path,bool report = true);
            ///< 'slurp' a file into a  memory string string
            std::string slurp (const std::string& path) ;
            /** \brief  given a subdir, search for all files with matching suffix in that directory, return back list of paths */
            vector<string> find(const string subdir,const string &suffix) ;
            /** given a wkt_reader (to help convert fro string to geo_xts_t), a lambda to convert from geo_id to geo_point, a subdir and a suffix,
             * read all files matching and provide them back as a shared pointer to a vector of \ref geo_xts_t */
            shared_ptr<vector<geo_xts_t>>
            load_from_directory(wkt_reader& wkt_io,function<ec::geo_point(int)> id_to_geo_point,const string& subdir,const string& suffix) ;
        }

        /** \brief the repository namespace have some simple classes that helps the orchestrator delegate io/config stuff.
         *
         */
        namespace repository {
            using namespace io;
            /** \brief a simple cell_repository, responsible for providing cells of supplied template parameter type C
             * with geo_cell_data from somewhere,
             * 'somewhere' is the files with specific names for the features:
             *  -# forest
             *  -# glacier
             *  -# lake (and reservoirs)
             *  -# reservoir mid-points (to figure out the reservoirs from the lakes)
             *  -# catchment map
             *  -# a digital terrain model (dtm)
             *  in a sub- directory .
             * These are read into memory \ref shyft::experimental::io::slurp and
             *  processed by \ref shyft::experimental::wkt_reader into boost::geometry multipolygons
             * then feed into the \ref geo_cell_data_computer that uses the rudimentary \ref region_grid
             * class to provide a grid-type of geometry, 1km x 1km squares.
             * \tparam C cell type that supports construction, state_t and parameter_t
             */
            template <class C>
            class cell_file_repository {
                string subdir;
                double x0;
                double y0;
                size_t nx;
                size_t ny;
                double dx;
                double dy;
                //vector<int> internal_to_catchment_id;
            public:
                typedef C cell_t;
                typedef typename C::state_t state_t;
                typedef typename C::parameter_t parameter_t;

                /**\brief construct a cell_file_repository that have its sources from the supplied sub_directory with the specified geometry
                 * \note that the geometry of dtm must match the dtm file
                 */
                cell_file_repository(string path,double x0,double y0,size_t nx,size_t ny,double dx=1000.0,double dy=1000.0)
                 :subdir(path),x0(x0),y0(y0),nx(nx),ny(ny),dx(dx),dy(dy) {}

                /** \brief read() does all needed stuff to get back a cell vector that can be used for region_model
                 */
                bool read(shared_ptr<vector<cell_t>> cells) {

                    // Step 1: get the files into maps/multi_polygons so that we can compute the cells.
                    wkt_reader wkt_io;
                    auto forests=wkt_io.read("forest",slurp(test_path(subdir+"/landtype_forest_wkt.txt")));
                    auto glaciers  =wkt_io.read("glacier",slurp(test_path(subdir+"/landtype_glacier_wkt.txt")));
                    auto lake_reservoirs  =wkt_io.read("lake",slurp(test_path(subdir+ "/landtype_lake_wkt.txt")));// nb! contains lakes AND reservoirs
                    auto rsv_points= wkt_io.read_points("rsv_mid_points",slurp(test_path(subdir+"/landtype_rsv_midpoints_wkt.txt")));// these points marks reservoirs from
                    auto dtmv=wkt_io.read_dtm("dtm",slurp(test_path(subdir+"/dtm_xtra.txt")));
                    auto catchment_map=wkt_io.read_catchment_map("catchment_map",slurp(test_path(subdir+"/catchments_wkt.txt")));
                    // ensure that the wkt_io operations went according to expectations
                    if(wkt_io.reported_errors.size()) {
                        cerr<<"cell_file_repository: Reported errors on test-data detected in wkt_reader:"<<endl;
                        for(auto e:wkt_io.reported_errors) {
                            cerr<<e<<endl;
                        }
                    }
                    // Step 2: establish the region_grid using the dtm data
                    region_grid rg(point_xy(x0,y0),nx,ny,dx,dy);//These matches exactly the dtm file in test data.
                    int misses=rg.set_dtm(dtmv);
                    if(misses>0)
                        throw runtime_error("cell_file_repository: the supplied digital terrain model file does not match with region-grid");

                    // Step 3. establish the multi_polygons for lakes and reservoirs
                    multi_polygon_ lakes=make_shared<multi_polygon>();
                    multi_polygon_ reservoirs=make_shared<multi_polygon>();
                    for(const auto& lr:*lake_reservoirs) {
                        polygon mpl;
                        geo::simplify(lr,mpl,300.0);// simplify, down to a x m. resolution, to gain speed
                        for(const auto& rp:*rsv_points) { // using the reservoir mid points to mark the lakes that are reservoirs
                            if(geo::covered_by(rp,lr)) {
                                reservoirs->push_back(mpl);
                            } else {
                                lakes->push_back(mpl);
                            }
                        }
                    }
                    // Step 4. Now with all data in place, compute the geo_cell_data for all catchments (keep relations)
                    geo_cell_data_computer region;
                    typedef vector<ec::geo_cell_data> geo_cell_data_vector;
                    typedef map<int,geo_cell_data_vector> catchment_gcd_map;
                    catchment_gcd_map  gcd_map;// a map between an internal catchment_id and the geo_cell_data_vector (by value could cost..)
                    const double default_radiation_factor=0.9;
                    for(const auto&kv:catchment_map) {
                        //internal_to_catchment_id.push_back(kv.first);// we create internal_to_catchment_id map, and  keep core internal id 0-based compact.
                        size_t internal_id = kv.first;//internal_to_catchment_id.size() - 1;
                        gcd_map.insert(
                            make_pair(kv.first,
                                region.catchment_geo_cell_data( // returns a vector<geo_cell_data>
                                      rg,// the region_grid providing cell(i,j) bounding box and elevation z
                                      internal_id,// all cells have this catchment_id
                                      default_radiation_factor,// all cells have this radiation factor (could be fetched from dtm ??)
                                      *kv.second,// the multi_polygon of this catchment
                                      *reservoirs,// features as multipolygons
                                      *lakes,
                                      *glaciers,
                                      *forests
                                )
                            )
                        );
                        // todo: just emplace_back catchment_cells to cells at this point ?
                        //       .. since we currently are not using catchment structure ?
                        // a: for now keep it like this, since we plan to use catchment structures later on
                    }
                    if(gcd_map.size()==0)
                        throw runtime_error("cell_file_repository: expected more than zero catchments in input data");
                    if(region.area_errors>0)
                        throw runtime_error("cell_file_repository: area_errors reported on geometry input data");
                    // Step 5. Finally, create the result as a vector of cell_t, with default state and parameters.
                    cells->clear();//auto cells= make_shared< vector<cell_t> >();
                    cells->reserve(rg.n_cells());
                    state_t cell_state;// need a state to fill in first time
                    cell_state.kirchner.q=100.0;
                    auto global_parameter= make_shared<shyft::core::pt_gs_k::parameter_t>(); // do we need an initial
                    for(const auto& kv:gcd_map) {
                        for(const auto& gcd:kv.second){
                            cells->emplace_back(cell_t{gcd,global_parameter,cell_state});
                        }
                    }
                    // this is how we could create a : region_model_t rm(*global_parameter,cells);
                    return cells->size()>0;
                }
            };

            /** \brief state_info, at least a time_stamp for which the state is valid, maybe some info/tags, and in
             * this early state also file_path to identify where the state is stored.
             * TODO: figure out how to replace file_path with some kind of id (template?)
             */
            struct state_info {
                ec::utctime time_stamp;
                string info;
                string file_path;
            };
            /**\brief a state repository based on file in a directory
             * \note currently just a skeleton to indicate the method signatures
             */
            template< class S>
            class state_file_repository {
                string path;
                size_t n_cells;
                public:
                state_file_repository(string path,size_t n_cells):path(path),n_cells(n_cells) {}
                //vector<state_info> find(ec::utcperiod within_period) {
                //    vector<state_info> r;
                //    return r;
                //}
                shared_ptr<vector<S>> read(string file_path){
                    auto r = make_shared< vector<S> >();
                    r->resize(n_cells);
                    return r;
                }
                //template < class S>
                //void write(const vector<S> & state_vector) {
                //}

            };

            /** \brief a time-series (ts) repository that provides geo-located time-series ready to
             * put into region_model/regional environment observation/forecast time-series.
             * Typically provides geo-located precipitation,temperature, radiation, windspeed,humidity, observed discharge from catchment
             */
            class geo_located_ts_file_repository {
                string path;
                //size_t n_cells;
              public:
                geo_located_ts_file_repository(string path):path(path){}
                /**\brief read from supplied path, geo-located ts, using the met-station id:position file to set geo-location
                 * \tparam RE regional environment structure keeping geo-located lists of precip etc.
                 * \param r the regional environment structure to be filled in.
                 * \note that the time-series type \ref geo_xts_t defined in shyft::experimental
                 */
                template<class RE>
                bool read(RE &r) {
                    wkt_reader wkt_io;
                    auto location_map=wkt_io.read_geo_point_map("met_stations",slurp(test_path(path+"/geo_point_map.txt")));
                    function<ec::geo_point(int)> geo_map;
                    geo_map=[&location_map] (int id) {return location_map[id].point;};
                    r.temperature= load_from_directory(wkt_io,geo_map,path,"temperature");
                    r.precipitation=load_from_directory(wkt_io,geo_map,path,"precipitation");
                    r.radiation=load_from_directory(wkt_io,geo_map,path,"radiation");
                    // ToDo: r.discharges=load_from_directory(wkt_io,geo_map,"neanidelv","discharge");
                    return wkt_io.reported_errors.size()==0;
                }
            };

        }

        namespace orchestration {
            using namespace repository;
            class region_orchestrator {
                // uses:
                // cell-repository : read_cells() -> shared_ptr<vector<cell_t>>
                // state-repository: find_state(criteria)-> id: meta-info(time,tag) ,read_state() -> shared_ptr<vector<state>> , write_state()
                // geo-ts-repository: read(period) -> region_environment
                // provides
                //  region_model with ref to cells
            };
        }

    }
}
