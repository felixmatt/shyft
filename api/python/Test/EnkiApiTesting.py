#!/usr/bin/env python
#Preconditions
# export LD_LIBRARY_PATH= path to enki.Os/bin/Debug
# PYTHONPATH contains enki.OS/bin/Debug (ro release)

import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt


enkiRootPath=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
enkiBinPath=os.path.join(enkiRootPath, 'bin', 'Release')
enkiModelPath=os.path.join(enkiRootPath, 'Data', 'SentralReg')
enkiModelPathTokke=os.path.join(enkiRootPath, 'Data', 'Tokke_Prod')
sys.path.insert(0, enkiBinPath)

import enki
import EnkiServiceNumPy as Enp

def TestNumPyMemoryAlloc():
    """ Verify zero memory leak when creating and disposing a lot of memory through std::vector
        and np arrays.
    """
    n=100000
    for i in xrange(0, 100):
        print 'Allocate loop ', i
        npa=  np.ones(n, np.float32)
        sta= Enp.FloatVector.FromNdArray(npa)
        print "Allocate loop, sz=", sta.size()
    return n

def TestUtcTimeVector():
    """ Just to verify that 64bits int array are convertible to utctime in enki.
    """
    llv=np.ones(10, np.int64)
    v=Enp.UtcTimeVector.FromNdArray(llv)
    print "Size of long long alias UtcTime vector is ", v.size()
    utc = enki.Calendar()
    Tnow= utc.trim(enki.utctimeNow(), enki.deltahours(1));
    v.push_back(Tnow)
    print "Size after adding Tnow to the list is  ", v.size()
    Xnow=v[10]
    print "Original timestamp is ", utc.toString(Tnow), " and roundtrip tstamp is ", utc.toString(Xnow)
    return

def TestBuildRegion():
    # How to create a scalar variable, and set/get values
    sv=enki.ScalarVariable("a", 3.14)
    sv.setValue(0, 0, 3.1415)
    vx=sv.getValue(0, 0)
    #-- how to create a network variable
    # a) create GeoPosition vector, with ID,X and Y positions
    geoPoints= enki.GeoPositionVector()
    geoPoints.push_back(enki.GeoPosition(1, 1.0, 1.0))
    geoPoints.push_back(enki.GeoPosition(2, 2.0, 2.0))
    # b) Create the network variable, with all points values set to 1.0
    nv= enki.NetworkVariable("nv1", geoPoints, 1.0)
    # c) Set values for points,
    nv.setValue(0, 0.5);
    nv.setValue(1, 2.7);
    #- How to create a rastervariable
    # a) Create a raster 2x2, with 2.22 values
    rst= enki.RasterFloat(2, 2, 2.22)
    # b) Create a rastervariable
    rv= enki.RasterVariable("rv1")
    rv.Initiate(rst)
    rv.GeoRef("utm32n", 0, 0, 2000, 2000)
    #- How to create a region
    # a) create a region with a name, reference system, and initial extents east - north
    region=enki.Region("ra", "utm32n", enki.GeoRect(0, 10000, 0, 10000))
    # b) add region variables
    region.AddVariable(sv)
    region.AddVariable(nv)
    region.AddVariable(rv)
    # c) Get the XML representing the region built (can be used to restore later)
    regionXml=region.asXmlString();
    region2= enki.Region.createFromXmlString(regionXml)
    # once we have a region, get out variables associated with the region
    regVar=region2.GetVariables()
    for v in regVar:
        print " '", v.name(), "' = ", v.getValue(0)
     
    region=None
    region2=None
    return 
 
def IncrementalTesting():
    utc = enki.Calendar()
    Tnow = utc.trim(enki.utctimeNow(), enki.deltahours(1));
    timeAxis = enki.FixedIntervalTimeAxis(Tnow, enki.deltahours(1), 24)
    FxSinus = enki.FxSinus(10.0, -5.0, 15.0, 10.0, Tnow, enki.deltahours(24))
    ts1 = enki.FxSinusTs(FxSinus, 'ts1', 1, enki.InputTs, timeAxis);
    
    print ts1.getValue(timeAxis.getTime(2))
    #print("Size of container is %s"%tscontainer.size())
    
    
    
    enkiService=enki.EnkiApiService()
    enkiService.setRootPath(enkiModelPath)
    runCfg=enki.RunConfiguration()
    runCfg.calculationModelName='enki'
    # Testing vector
    tv= enki.UtcTimeVector()
    tv.push_back(Tnow)
    qv= enki.IntVector()
    qv.push_back(0);qv.push_back(0);qv.push_back(0);
    pvv= np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    #print "Now it works..:", Ex.rms(pvv)
    #xvv=enki.MakeFloatVectorFromNp(pvv)
    #print xvv.size(), " xvv " , xvv[0] 
    vv= enki.FloatVector()
    vv.push_back(1.0)
    vv.push_back(2.0)
    vv.push_back(3.0)
    vv.push_back(4.0)
    npvv=np.fromiter(vv, np.float32)
    print "Ok, got np array:", npvv
    #xvv=Ex.MakeFloatVector(npvv)
    xxv=Enp.FloatVector.FromNdArray(pvv)
    print "Ok, now a float vector,", xxv.size()
    #--  now check float64 FloatVector
    dv= np.ones(3, dtype=np.float64);
    fdv= Enp.FloatVector.FromNdArrayDouble(dv);
    print "Ok, now a float vector from np.float64", fdv.size()
    for i in range(fdv.size()):
        print i, "=", fdv[i]


    #--  xx
    rst1=enki.RasterFloat(2, 2, xxv);
    print "Raster from flat float array ", rst1.nRows(), " x ", rst1.nCols(), "=", rst1.nCells()
    #xnpvv=enki.MakeFloatVectorFromNp(ct.c_void_p(npvv.ctypes.data),ct.c_int(3))
    #print "Round trip ",xnpvv.size()
    qv= enki.IntVector()
    qv.push_back(0);qv.push_back(0);qv.push_back(0);
    factory= enki.TsFactory();
    metaInfo=enki.MetaInfo(1, "InstantRunoff", "InstantRunoff", 0, enki.InputTs)
    pts= factory.CreatePointTs(metaInfo, 3, Tnow, enki.deltahours(1), vv, qv);
    print "tv size=", tv.size(), " vv.size", vv.size(), " pts.Count()=", pts.nPoints()
    for i in range(pts.nPoints()):
        ti=pts.getTime(i)
        print i, utc.toString(ti), pts.getValue(i)

    runCfg.regionName='SentralReg'
    runCfg.Tstart = utc.time(enki.YMDhms(2012, 01, 01, 00, 00, 00))
    runCfg.deltaT=enki.deltahours(1)
    runCfg.nSteps=24
    runCfg.runType=enki.UpdateRun
    #runCfg.ts=tscontainer.tsc;
    #tscontainer.tsc=runCfg.ts #tsc2=TsContainer(runCfg.ts);
    #tsx=tscontainer.item(0);
    #for i in range(timeAxis.size()):
    #    ti=timeAxis.getTime(i)
    #    print i,utc.toString(ti), tsx.getValue(i),' .. ', FxSinus(ti)

class source_info(object):

    def __init__(self, network_name=None, source_type=None, vector_type=None, ts_mapping=None):
        self.name = network_name
        self.vector_type = vector_type
        self.source_type = source_type
        self.ts_mapping = ts_mapping
        assert network_name != None and vector_type != None and source_type != None and ts_mapping != None
        
    def timeseries_map(self, network_index):
        return self.ts_mapping[network_index]
                
class EnkiXmlTest(object):

    def __init__(self, region="Tokke", plot_results=True):
        self.region_name = region
        if region == "Tokke":
            self.state_name = "Tokke.state.2014-08-20_06"
        elif region == "CentralRegion":
            self.state_name = "CentralRegion.state.2014-09-01_02"
        else:
            raise RuntimeError("Don't know the region {}".format(region))
        self.plot_results = plot_results
        self.cal = enki.Calendar()
        self.temperature_sources = enki.TemperatureSourceVector()
        self.temperature_index_map = dict()
        self.temperature_ts_map = dict() # fill from runconfiguration [ts.enki_id]=ts
        self.cell_map = []
        self.catchment_map=[]
        self.ncols = 0
        self.crows = 0
    
    def enki_var_type_to_string(self, vt):
        """ convert enki variable type to readable string
        """
        if vt == enki.ScalarType: return "Scalar"
        if vt == enki.PointType: return "Network"
        if vt == enki.RasterType: return "Grid"
        raise ValueError("Unknown variable type: {}, could not convert".format(vt))

    
    def extract_point_source(self, variables, source_info): # sourceName, sourceType, tsForNetworkId, sourceVector):
        for v in variables:
            if v.variableType() == enki.PointType:
                if v.name() == source_info.name:
                    result = source_info.vector_type()
                    for n in xrange(v.nColumns()):
                        geoposition = v.geoPosition(0, n)
                        z = v.getValue(n)
                        src = source_info.source_type() #enki.TemperatureSource()
                        src.geopoint = enki.GeoPoint(geoposition.X, geoposition.Y, z)
                        src.ts = source_info.timeseries_map(geoposition.Id) #self.getTemperatureTsForNetworkId(geoposition.Id)
                        result.append(src)
        return result

    def extract_raster(self, variables, name): # sourceName, sourceType, tsForNetworkId, sourceVector):
        for v in variables:
            if v.variableType() == enki.RasterType:
                if v.name() == name:
                    return v
        raise ValueError("Could not extract raster with name '{}'".format(name))

    def extract_scalar(self, variables, name):
        for v in variables:
            if v.variableType() == enki.ScalarType:
                if v.name() == name:
                    return v
        raise ValueError("Could not extract scalar value with name '{}', giving up.".format(name))
    
    def print_variable(self, v):
        vt = self.enki_var_type_to_string(v.variableType())
        georefsys = v.geoRefSystem()
        if vt == "Scalar":
            print "{:16} {} = {}".format(v.name()+":", vt, v.getValue(0))
        elif vt == "Network":
            print "{}:".format(v.name())
            for n in xrange(v.nColumns()):
                geopoint = v.geoPosition(0, n)
                print " {:3} ({:08.4f}, {:08.4f}) [{:08.3f}]".format(str(geopoint.Id) + ":", geopoint.X, geopoint.Y, v.getValue(n))
        else:
            gr = v.geoRect();
            print "{:16} {} ({}, {}, {}, {}, {}) {}x{}".format(v.name() + ":", vt, georefsys, gr.west, gr.east,\
                    gr.south, gr.north, v.nColumns(), v.nRows())

    @property
    def region(self):
        rname = self.region_name
        with open("../../../Data/{}/{}.rgx".format(rname, rname), "rb") as region_file:
            region_xml = region_file.read()
        return enki.Region.createFromXmlString(region_xml)

    @property
    def model_state(self):
        rname = self.region_name
        sname = self.state_name
        with open('../../../Data/{}/State/{}.stx'.format(rname, sname), 'rb') as state_file:
            state_xml = state_file.read()
        return enki.ModelState.createFromXmlString(state_xml)
    
    def extract_point_source_variables_from_region(self, region, t0=0):
        region_var = region.GetVariables()

        delta_t = enki.deltahours(1)
        time_axis = enki.FixedIntervalTimeAxis(t0, delta_t, 10*24 - 1) # hourly data for 100 days

        class MockSourceMap(dict):

            def __init__(s, f_0=0.0, f_min=0.0, f_max=1.0, amp=1.0, delay=enki.deltahours(9), period_length=enki.deltahours(24)):
                s.f_0 = f_0
                s.f_min = f_min
                s.f_max = f_max
                s.amp = amp
                s.delay = delay
                s.period_length = period_length

            def __getitem__(s, idx):
                t_w = t0
                ts_name = "".join([random.choice("abcdefghijklmnop") for _ in xrange(10)])
                f = enki.FxSinus(s.f_0, s.f_min, s.f_max, s.amp, t_w + s.delay, s.period_length)
                ts = enki.FxSinusTs(f, ts_name, idx, enki.InputTs, time_axis)
                return ts

        temperature_sources = self.extract_point_source(region_var,
                                                        source_info('Tstats_elev', 
                                                                    enki.TemperatureSource,
                                                                    enki.TemperatureSourceVector,
                                                                    MockSourceMap(4.0, -2.0, 10.0, 6.0, 
                                                                                  enki.deltahours(9), enki.deltahours(24))))
        precipitation_sources = self.extract_point_source(region_var,
                                                          source_info('Pstats_elev',
                                                                      enki.PrecipitationSource,
                                                                      enki.PrecipitationSourceVector, 
                                                                      MockSourceMap(-10.0, 0.0, 4.00, 100.0,
                                                                                    0, enki.deltahours(30))))
        radiation_sources = self.extract_point_source(region_var,
                                                      source_info('Rstats',
                                                                  enki.RadiationSource,
                                                                  enki.RadiationSourceVector,
                                                                  MockSourceMap(0.0, 0.0, 400.0, 700.0, 
                                                                                enki.deltahours(7), enki.deltahours(24))))

        return {'temperature': temperature_sources, 
                'precipitation': precipitation_sources,
                'radiation': radiation_sources}, time_axis

    def extract_ptgsk_params_from_region(self, region):
        region_var = region.GetVariables()

        e = lambda v: self.extract_scalar(region_var, v).getValue(0)

        pt_names = ("LandAlbedo", "PTalpha")
        gs_names = ("Lastwinterday", "Tx", "Windscale", "Windconst", "MaxLWC", "SurfaceLayer",
                    "MaxAlbedo", "Minalbedo", "FastDecayRate", "SlowDecayrate", "ResetSnowDepth", "GlacierAlb")
        gs_args = [e(n) for n in gs_names]
        gs_args[0] = int(round(gs_args[0]))

        ae_names = ("EvapQscale",)
        k_names = ("lnTau3", "DlnTauDlnQ") # One could set c1, c2, and c3 here, since I cannot find them in the data set.

        pt_params = enki.PriestleyTaylorParameter(*[e(n) for n in pt_names])
        gs_params = enki.GammaSnowParameter(*gs_args)
        ae_params = enki.ActualEvapotranspirationParameter(*[e(n) for n in ae_names])
        g_params = enki.GridParameter()
        k_arg_init = [e(n) for n in k_names]
        c = [0.0, 0.0, 0.0]
        c[1] = -k_arg_init[1]
        c[0] = 3*c[1] - k_arg_init[0]
        k_params = enki.KirchnerParameter(*c)

        return enki.PTGSKParam(pt_params, gs_params, ae_params, k_params, g_params)

    def extract_state_rasters(self, var, pt_state_names, gs_state_names, k_state_names):
        return {'pt': {key: value for (key, value) in ((name, self.extract_raster(var, name)) for name in pt_state_names)},
                'gs': {key: value for (key, value) in ((name, self.extract_raster(var, name)) for name in gs_state_names)},
                'k': {key: value for (key, value) in ((name, self.extract_raster(var, name)) for name in k_state_names)}}

    def build_ptgsk_state_from_raster_idx(self, rasters, idx, pt_state_names, gs_state_names, k_state_names):
        e = lambda m, n: rasters[m][n].getValue(*idx)
        pt_state = enki.PriestleyTaylorState(*[e('pt', n) for n in pt_state_names])
        gs_state = enki.GammaSnowState(*[e('gs', n) for n in gs_state_names])
        k_state = enki.KirchnerState(*[e('k', n) for n in k_state_names])
        return enki.PTGSKStat(pt_state, gs_state, k_state)

    @staticmethod
    def transform_kirchner_parameters(ln_tau_3, d_ln_tau_d_ln_q):
        c2 = -d_ln_tau_d_ln_q
        c1 = 3*c2 - ln_tau_3
        c3 = 0.0
        return c1, c2, c3


    def extract_grid_cells_from_region(self, region, model_state, cell_type=enki.CollectingPTGSKGridCell):
        pt_state_names = ()
        gs_state_names = ("Albedo", "SnowLiqWatDepth", "SurfEnergy",
                          "SDCShape", "SDC_M", "AccMeltDepth", "IsoPotEnergy", "TempSWE")
        k_state_names = ("InstantRunoff",)

        region_var = region.GetVariables()
        elevation = self.extract_raster(region_var, 'Elevation')
        catchment = self.extract_raster(region_var, 'Catchments')
        landtype = self.extract_raster(region_var, 'Landuse')
        glacier_frac = self.extract_raster(region_var, 'GLACIERS')
        init_bare_ground_frac = self.extract_raster(region_var, "Inity")
        snow_cv = self.extract_raster(region_var, "SDC_CV")
        grid_rscale = self.extract_raster(region_var, "Grid_Rscale")
        grid_ln_tau_3 = self.extract_raster(region_var, "Grid_lnTau3")
        grid_d_ln_tau_d_ln_q = self.extract_raster(region_var, "Grid_dlnTaudlnQ")

        #radiation_slope_factor = 0.9
        #.. combine into destination cells.
        cells = cell_type.vector_t()
        cells.reserve(catchment.nRows()*catchment.nColumns())
        state_rasters = self.extract_state_rasters(model_state.GetVariables(), pt_state_names, gs_state_names, k_state_names)
        catchment_map = []
        self.nrows = catchment.nRows()
        self.ncols = catchment.nColumns()
        for r in xrange(catchment.nRows()):
            for c in xrange(catchment.nColumns()):
                catchment_id = int(round(catchment.getValue(r, c)))
                if not catchment_id == 0: #is_flagged(r, c):
                    self.cell_map.append((r, c)) # For plotting
                    z = elevation.getValue(r, c)
                    geopos = elevation.geoPosition(r, c)
                    loc = enki.GeoPoint(geopos.X, geopos.Y, z)
                    if not catchment_id in catchment_map:
                        catchment_map.append(catchment_id)
                        mapped_catchment_id = len(catchment_map) - 1
                    else:
                        mapped_catchment_id = catchment_map.index(catchment_id)

                    c1, c2, c3 = self.transform_kirchner_parameters(grid_ln_tau_3.getValue(r, c), grid_d_ln_tau_d_ln_q.getValue(r, c))

                    # Note that we operate with an enum without the flag value in the new core, hence the -1 on the land type.
                    #cell = cell_type(loc, mapped_catchment_id,
                    #                 landtype(loc) - 1, glacier_frac(loc),
                    #                 init_bare_ground_frac(loc), snow_cv(loc),
                    #                 grid_rscale(loc), c1, c2, c3, elevation.geoArea(loc))

                    cell = cell_type(loc, mapped_catchment_id,
                                     int(round(landtype.getValue(r, c))) - 1, glacier_frac.getValue(r, c),
                                     init_bare_ground_frac.getValue(r, c), snow_cv.getValue(r, c),
                                     grid_rscale.getValue(r, c), c1, c2, c3, elevation.geoArea(r, c))
                    cell.set_state(self.build_ptgsk_state_from_raster_idx(state_rasters, (r, c),
                                                                          pt_state_names, gs_state_names, k_state_names))
                    cells.append(cell)
        self.catchment_map = catchment_map
        return cells, catchment_map

    def catchment_list(self, region):
        catchment = self.extract_raster(region.GetVariables(), 'Catchments')
        catchment_list = []
        for r in xrange(catchment.nRows()):
            for c in xrange(catchment.nColumns()):
                cid = catchment.getValue(r, c)
                if cid not in catchment_list:
                    catchment_list.append(int(round(cid)))
        return catchment_list


    def extract_grid_cells_for_catchment(self, region, model_state, catchment_id, cell_type=enki.OptimizingPTGSKGridCell):
        pt_state_names = ()
        gs_state_names = ("Albedo", "SnowLiqWatDepth", "SurfEnergy",
                          "SDCShape", "SDC_M", "AccMeltDepth", "IsoPotEnergy", "TempSWE")
        k_state_names = ("InstantRunoff",)

        region_var = region.GetVariables()
        elevation = self.extract_raster(region_var, 'Elevation')
        catchment = self.extract_raster(region_var, 'Catchments')
        landtype = self.extract_raster(region_var, 'Landuse')
        glacier_frac = self.extract_raster(region_var, 'GLACIERS')
        init_bare_ground_frac = self.extract_raster(region_var, "Inity")
        snow_cv = self.extract_raster(region_var, "SDC_CV")
        grid_rscale = self.extract_raster(region_var, "Grid_Rscale")
        grid_ln_tau_3 = self.extract_raster(region_var, "Grid_lnTau3")
        grid_d_ln_tau_d_ln_q = self.extract_raster(region_var, "Grid_dlnTaudlnQ")

        #radiation_slope_factor = 0.9
        #.. combine into destination cells.
        destinations = cell_type.vector_t()
        destinations.reserve(catchment.nRows()*catchment.nColumns())
        state_rasters = self.extract_state_rasters(model_state.GetVariables(), pt_state_names, gs_state_names, k_state_names)
        self.nrows = catchment.nRows()
        self.ncols = catchment.nColumns()
        for r in xrange(catchment.nRows()):
            for c in xrange(catchment.nColumns()):
                cid = int(round(catchment.getValue(r, c)))
                if cid != catchment_id:
                    continue
                self.cell_map.append((r, c)) # For plotting
                z = elevation.getValue(r, c)
                geopos = elevation.geoPosition(r, c)
                geopoint = enki.GeoPoint(geopos.X, geopos.Y, z)

                c1, c2, c3 = self.transform_kirchner_parameters(grid_ln_tau_3.getValue(r, c), grid_d_ln_tau_d_ln_q.getValue(r, c))

                # Note that we operate with an enum without the flag value in the new core, hence the -1 on the land type.
                dest = cell_type(geopoint, 0,
                                 int(landtype.getValue(r, c) - 1), glacier_frac.getValue(r, c),
                                 init_bare_ground_frac.getValue(r, c), snow_cv.getValue(r, c),
                                 grid_rscale.getValue(r, c), c1, c2, c3, elevation.geoArea(r, c))
                dest.set_state(self.build_ptgsk_state_from_raster_idx(state_rasters, (r, c),
                                                                      pt_state_names, gs_state_names, k_state_names))
                destinations.append(dest)
        self.catchment_map = catchment_map = {0: catchment_id}
        return destinations, catchment_map

    def extract_interpolation_parameters_from_region(self, region):
        region_var = region.GetVariables()
        e = lambda n: self.extract_scalar(region_var, n).getValue(0)
        btk_p_names = ("PriSDtgrad", "Tsill", "Tnugget", "Trange", "Tzscale")
        btk_param = enki.BTKParameter(-0.6, *[e(n) for n in btk_p_names])
        idw_p_names = ("MaxIntStats", "MaxIntDist")

        args = [e(n) for n in idw_p_names]
        args[0] = int(round(args[0]))
        prec_p_names = ("PrecGrad")
        prec_param = enki.IDWPrecipitationParameter(e("PrecGrad"), *args)
        ws_param = enki.IDWParameter(*args)
        rad_param = enki.IDWParameter(*args)
        rel_hum_param = enki.IDWParameter(*args)
        return enki.InterpolationParameter(btk_param, prec_param, ws_param, rad_param, rel_hum_param)
    
    def extract_model_parameters_from_region(self, region):
        """ from the region, extract some constants that are used for humidity and windspeed """
        region_var=region.GetVariables()
        return (self.extract_scalar(region_var, 'ConstWind').getValue(0), 0.01*self.extract_scalar(region_var, 'ConstHum').getValue(0))

    def build_model_for_region(self, region, model_state):
        interpolation_parameter = self.extract_interpolation_parameters_from_region(region)
        global_ptgsk_parameter = self.extract_ptgsk_params_from_region(region)
        destinations, catchment_map = self.extract_grid_cells_from_region(region, model_state)
        model = enki.FullPTGSKModel(len(catchment_map), 2.0, 0.70)
        sources, source_time_axis = self.extract_point_source_variables_from_region(region, model_state.timestamp())
        time_axis = enki.FixedIntervalTimeAxis(source_time_axis.start(), source_time_axis.delta(), source_time_axis.size())
        model.initialize(interpolation_parameter, global_ptgsk_parameter, time_axis, destinations,
                         sources['temperature'],
                         sources['precipitation'],
                         sources['radiation'], None, None)
        return model

    def test_build_model(self):
        model = self.build_model_for_region(self.region)
        print "Model created and initialized"

    def plot_raster_data(self, destinations, *plot_data):
        from matplotlib import colors
        palette = plt.cm.coolwarm
        #x = np.array([d.geoPoint().x for d in destinations])
        #y = np.array([d.geoPoint().y for d in destinations])
        x = np.array([i*1000 for i in xrange(self.ncols)])
        y = np.array([-i*1000 for i in xrange(self.nrows)])
        p_arr = np.ma.array(np.zeros((self.nrows, self.ncols), dtype='d'), mask=True) 
        p_arr.shape = self.nrows, self.ncols
        for i, pd in enumerate(plot_data):
            label, cb, y_min_default, y_max_default = pd
            data = np.array([cb(d) for d in destinations])
            
            for j, d in enumerate(data):
                p_arr[self.cell_map[j]] = d

            plt.subplot(1, len(plot_data), i + 1)
            #plt.scatter(x, y, s=20, c=data, marker='o')
            y_min=min(data)
            y_max=max(data)
            if abs(y_min-y_max) < 0.001:
                y_min=y_min_default
                y_max=y_max_default

            plt.imshow(1.0*p_arr,
                            interpolation='nearest', 
                            cmap=palette,
                            norm=colors.Normalize(vmin=y_min, vmax=y_max, clip=False))
            #plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.colorbar(shrink=0.4)
            plt.ylabel(label)
        plt.show()

    def plot_catchment_results(self, model):
        plt.hold(True)
        catchment_res = model.get_catchment_results()
        nplots = 6
        n_c = len(self.catchment_map)

        def plot_data(data):
            [plt.plot([_ for _ in d]) for d in data]

        plt.subplot(nplots, 1, 1)
        plot_data(catchment_res)
        plt.legend(["Disch[m3/s]: {}".format(i) for i in xrange(n_c)])
        plt.title("Hourly discharge for catchments in m^3/sec")

        plt.subplot(nplots, 1, 2)
        plot_data([model.get_catchment_precipitation(i) for i in xrange(n_c)])
        plt.legend(["prec: {}".format(i) for i in xrange(n_c)])
        plt.title("Avg Precipitation[mm/h]")

        plt.subplot(nplots, 1, 3)
        plot_data([model.get_catchment_temperature(i) for i in xrange(n_c)])
        plt.legend(["temp: {}".format(i) for i in xrange(n_c)])
        plt.title("Avg temp[C]")

        plt.subplot(nplots, 1, 4)
        plot_data([model.get_catchment_snow_swe(i) for i in xrange(n_c)])
        plt.legend(["snow_swe: {}".format(i) for i in xrange(n_c)])
        plt.title("Snow storage SWE[mm]")

        plt.subplot(nplots, 1, 5)
        plot_data([model.get_catchment_snow_output(i) for i in xrange(n_c)])
        plt.legend(["snow_output: {}".format(i) for i in xrange(n_c)])
        plt.title("Output from Gamma Snow [mm/h]")

        plt.subplot(nplots, 1, 6)
        plot_data([model.get_catchment_pe_output(i) for i in xrange(n_c)])
        plot_data([model.get_catchment_ae_output(i) for i in xrange(n_c)])
        plt.legend(["Pot evap: {}".format(i) for i in xrange(n_c)] + ["Act evap: {}".format(i) for i in xrange(n_c)])
        plt.title("Output from Pot and act evap [mm/h]")

        plt.show()

    def test_build_and_run_model(self):
        model = self.build_model_for_region(self.region, self.model_state)
        idx = 12
        cbs = [("Temperature  (+12h)", lambda d: d.temperature[idx], 0, 0),
               ("Precipitation(+12h)", lambda d: d.precipitation[idx], 0, 0),
               ("Radiation    (+12h)", lambda d: d.radiation[idx], 0, 500),
               ("Catchment_id", lambda d: d.catchment_id, 0, 0)
              ]
        if self.plot_results:
            self.plot_raster_data(model.get_grid_cells(), *cbs)
        model.run()
        if self.plot_results:
            self.plot_catchment_results(model)
            plot_data = [("q_avg", lambda d: d.response.kirchner.q_avg, 0, 0), ("Snow storage", lambda d: d.response.gs.storage, 0, 0)]
            self.plot_raster_data(model.get_grid_cells(), *plot_data)

    def make_fake_target(self, catchment_number, region, model_state):
        interpolation_parameter = self.extract_interpolation_parameters_from_region(region)
        global_ptgsk_parameter = self.extract_ptgsk_params_from_region(region)
        destinations, catchment_map = self.extract_grid_cells_for_catchment(region, model_state, catchment_number)
        sources, source_time_axis = self.extract_point_source_variables_from_region(region, model_state.timestamp())
        time_axis = enki.FixedIntervalTimeAxis(source_time_axis.start(), source_time_axis.delta(), source_time_axis.size())

        true_model = enki.ReducedPTGSKModel(len(catchment_map), 2.0, 0.70)
        true_model.initialize(interpolation_parameter, global_ptgsk_parameter, time_axis, destinations,
                              sources['temperature'],
                              sources['precipitation'],
                              sources['radiation'], None, None)
        true_model.run()
        catchment_params = true_model.get_grid_cells()[0].local_parameter()
        return true_model.get_catchment_results()[0], catchment_params


    def test_optimize(self):
        region = self.region
        catchment_index = 2
        print "Optimizing for catchment {}".format(self.catchment_list(region)[catchment_index])
        catchment_number = self.catchment_list(region)[catchment_index]
        model_state = self.model_state
        interpolation_parameter = self.extract_interpolation_parameters_from_region(region)
        global_ptgsk_parameter = self.extract_ptgsk_params_from_region(region)
        destinations, catchment_map = self.extract_grid_cells_for_catchment(region, model_state, catchment_number) 
        sources, source_time_axis = self.extract_point_source_variables_from_region(region, model_state.timestamp())
        time_axis = enki.FixedIntervalTimeAxis(source_time_axis.start(), source_time_axis.delta(), source_time_axis.size())

        target, params = self.make_fake_target(catchment_number, region, model_state)

        search_model = enki.ReducedPTGSKModel(1, 2.0, 0.70)
        search_model.initialize(interpolation_parameter, global_ptgsk_parameter, time_axis, destinations,
                                sources['temperature'],
                                sources['precipitation'],
                                sources['radiation'], None, None)


        op = params.kirchner
        p_min = [0.7*x for x in (op.c1(), op.c2())] + [-0.001]
        p_max = [1.2*x for x in (op.c1(), op.c2())] + [ 0.001]
        import random
        random.seed(1)
        p_init = [random.uniform(p_min[i], p_max[i]) for i in xrange(len(p_min))];
         
        print "True parameter = {}".format([op.c1(), op.c2(), op.c3()])
        print "Initial guess = {}".format(p_init)
        kirchner_opt = enki.PTGSKKirchnerOptModel(search_model, 0, target, time_axis, p_min, p_max)
        p_first = kirchner_opt.optimize(p_init, 0.3, 1.0e-5)
        print "First solution = {}".format(p_first)
        print "Second solution = {}".format(kirchner_opt.optimize(p_first, 0.1, 1.0e-6/4.0))


    def test_optimize_state_reset(self):
        region = self.region
        model_state = self.model_state

        catchment_id = 2
        target, params = self.make_fake_target(catchment_id, region, model_state)
        print "got target"

        interpolation_parameter = self.extract_interpolation_parameters_from_region(region)
        global_ptgsk_parameter = self.extract_ptgsk_params_from_region(region)
        destinations, catchment_map = self.extract_grid_cells_from_region(region, model_state, cell_type=enki.OptimizingPTGSKGridCell)
        sources, source_time_axis = self.extract_point_source_variables_from_region(region, model_state.timestamp())
        time_axis = enki.FixedIntervalTimeAxis(source_time_axis.start(), source_time_axis.delta(), source_time_axis.size())

        search_model = enki.ReducedPTGSKModel(len(catchment_map), 2.0, 0.70)
        search_model.initialize(interpolation_parameter, global_ptgsk_parameter, time_axis, destinations,
                                sources['temperature'],
                                sources['precipitation'],
                                sources['radiation'], None, None)

        op = params.kirchner
        p = [op.c1(), op.c2(), op.c3()]
        p_min = [0.7*x for x in (op.c1(), op.c2())] + [-0.001]
        p_max = [1.2*x for x in (op.c1(), op.c2())] + [ 0.001]

        print "Construct optimizer"
        print p_min
        print p_max

        kirchner_opt = enki.PTGSKKirchnerOptModel(search_model, catchment_id, target, time_axis, p_min, p_max)
        print "Done"

        print "Should be zero: {} ".format(kirchner_opt.run(p) - kirchner_opt.run(p))


    def test_extract_interpolation_parameters_from_region(self):
        return self.extract_interpolation_parameters_from_region(self.region)
             
    def test_extract_grid_cells_from_region(self):
        return self.extract_grid_cells_from_region(self.region)
             
    def print_region_variables_from_region_xml(self):
        region_var = self.region.GetVariables()            
        print "Generic print"
        for rv in region_var:
            self.print_variable(rv)               

    def print_scalars_from_region_xml(self):
        region_var = self.region.GetVariables()            
        print "Scalars:"
        for rv in region_var:
            if rv.variableType() == enki.ScalarType:
                self.print_variable(rv)

    def extract_state_variables_from_state(self):
        model_state = self.model_state
        state_var = model_state.GetVariables()
        state_time = model_state.timestamp()
        print "State timestamp:", self.cal.toString(state_time)
        for sv in state_var:
            self.print_variable(sv)

     
class EnkiTest(object):
    def __init__(self):
        self.cal = enki.Calendar();

    
    def createRunConfiguration(self, regionName, modelName, runType, Tstart, deltaT, nSteps, dataOffset):
        r=enki.RunConfiguration()
        r.regionName=regionName
        with open('../../../Data/SentralReg/'+regionName+'.rgx', 'rb') as regionFile:
            r.regionXml= regionFile.read()
        r.modelName=modelName
        with open('../../../Data/SentralReg/'+modelName+'.mdx', 'rb') as modelFile:
            r.modelXml= modelFile.read()
        dt= self.cal.calendarUnits(Tstart);
        stateFileName= "%s.state.%04d-%02d-%02d_%02d.stx"%(modelName, dt.year, dt.month, dt.day, dt.hour)
        with open('../../../Data/SentralReg/'+stateFileName) as stateFile:
            r.stateXml= stateFile.read()
        r.Tstart=Tstart
        r.deltaT=deltaT
        r.nSteps=nSteps
        r.runType=runType
        Tw = self.cal.time(enki.YMDhms(2012, 3, 1, 0, 0, 0))
        Td= r.Tstart + dataOffset
        inputTsType= enki.InputTs
        timeAxis=enki.FixedIntervalTimeAxis(Td, r.deltaT, r.nSteps)
        # now we have to put at least one timeseries in there..:
        # temperature: dayly sinus curve with average +3.0, +-8 deg C, phase +9 hours, (mid day at 1500)
        # the node  average= 3.0 + i/numberofNodes. (4.0 at highest node)
        nTStatsNodes=14
        TnodeId= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]
        #tsc= enki.TsContainer()
        for i in range(nTStatsNodes):
            fx = enki.FxSinus(3.0 + i/(1.0*nTStatsNodes), -10.0, +20.0, 8.0, Tw+enki.deltahours(9), enki.deltahours(24))
            tstats= enki.FxSinusTs(fx, "TStats", TnodeId[i], inputTsType, timeAxis)
            r.ts.push_back(tstats);
        
        # precipitation:
        #  sine curve , with period equal to forecast length, clipped to zero
        
        nPstatsNodes=10
        forecastLength=r.deltaT*r.nSteps;
        for i in range(nPstatsNodes):
            fx= enki.FxSinus(10.0 + i/(1.0*nPstatsNodes), 0.0, +20.0, 8.0, Tw, forecastLength)
            pstats= enki.FxSinusTs(fx, "PStats", 1 + i, inputTsType, timeAxis)
            r.ts.push_back(pstats)
        
        # radiation
        #  max at  high day,  1200, aprox 400, zero during dark.
        # uses a sin wave amplitude 700,clipped at 400 and 0. starts at 0700
        nRstatNodes = 1 #// 1,2,4,5
        RnodeId=[1, 2, 4, 5]
        for i in range(nRstatNodes):
            fx= enki.FxSinus(0.0, 0.0, +400.0, 500.0, Tw+ enki.deltahours(7), enki.deltahours(24))
            rstats= enki.FxSinusTs(fx, "RStats", RnodeId[i], inputTsType, timeAxis)
            r.ts.push_back(rstats)
       
        simplSimDischarge= enki.MetaInfo(0, "SimplSimDischarge", "SimplSimDischarge", 0, enki.OutputTs)
        instantRunoff= enki.MetaInfo(0, "InstantRunoff", "InstantRunoff", 0, enki.OutputTs)
        r.resultVariables.push_back(simplSimDischarge)
        r.resultVariables.push_back(instantRunoff)
        return r 
    
    def run(self, runType, Tstart, nSteps, dataOffset):
        runCfg = self.createRunConfiguration('SentralReg', 'enki', runType, Tstart, enki.deltahours(1), nSteps, dataOffset)
        runCfg.modelName = 'enki'
        enkiService = enki.EnkiApiService(False)
        enkiService.setRootPath(enkiModelPath)
        print "Running enki .."
        result=enkiService.run(runCfg)
        print "Done,", result.success
        return result



    def demo(self):
        Tstart= self.cal.time(enki.YMDhms(2012, 3, 8, 0, 0, 0))
        nSteps=12
        runCfg=self.createRunConfiguration('SentralReg', 'Enki', enki.UpdateRun, Tstart, enki.deltahours(1), nSteps, enki.deltahours(0))
        runCfg.calculationModelName='Enki'
        print runCfg.regionName
        print runCfg.calculationModelName
        print self.cal.toString(runCfg.Tstart), runCfg.deltaT, runCfg.nSteps
        result = self.run(enki.UpdateRun, Tstart, nSteps, enki.deltahours(0))
        
        print result.messageCount(), result.tsCount()
        print 'The stateXml size=', len(result.stateXml)
        
        for i in range(result.messageCount()):
            try:
                if(len(result.message(i))) :
                    print result.message(i)
            except:
                pass
        
        for i in range(result.tsCount()):
            ts_i= result.ts(i)
            try:
                print ts_i.getName(), ts_i.getEnkiName(), ts_i.id()
            except:
                pass
            for j in range (ts_i.nPoints()):
                try:
                    print i, self.cal.toString(ts_i.getTime(j)), ' ', ts_i.getValue(j)
                except:
                    pass
                
        for i in range(result.rtsCount()):
            rts_i = result.rts(i)
            for j in range (rts_i.nPoints()):
                rts=rts_i.getValue(j)
                asVector=rts.asRowMajorVector()
                try:
                    print rts_i.getName(), '/', rts_i.getEnkiName(), rts_i.id(), i, self.cal.toString(rts_i.getTime(j)), 'rxc=', rts.nRows(), ' x ', rts.nCols()
                except:
                    pass
            
        
        print 'Done demo run' 

if __name__ == "__main__":
    print 'Starting demo'
    #try:
        
    #TestUtcTimeVector()
    #IncrementalTesting()
    #TestBuildRegion()
    #print 'Test mempory alloc'
    #TestNumPyMemoryAlloc()
    #.. lines below are for experimenting with datasets at statkraft only, not distributed
    #et.demoTokke()    
    enki_xml_test = EnkiXmlTest("CentralRegion", plot_results=True)
    #enki_xml_test = EnkiXmlTest("Tokke")
    #enki_xml_test.print_region_variables_from_region_xml()
    #enki_xml_test.extract_point_source_variables_from_region(enki_xml_test.region)
    #dest = enki_xml_test.test_extract_grid_cells_from_region()
    #enki_xml_test.test_extract_interpolation_parameters_from_region()
    #print "Test destinations, number of destinations = {}".format(len(dest))

    #enki_xml_test.extract_state_variables_from_state()
    #enki_xml_test.catchment_list(enki_xml_test.region)
    enki_xml_test.test_build_and_run_model()

    #enki_xml_test.test_optimize()
    #enki_xml_test.test_optimize_state_reset()
        
    #et= EnkiTest()
    #et.demo()
    print "All tests ran fine"
    #except :
        #print "Unexpected error:", sys.exc_info()[0]
    #finally:
    #    print "Exit.."
