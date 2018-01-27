import matplotlib.pylab as plt
from patch import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import cm
from os import path
from shyft.repository.service.gis_location_service import GisLocationService
from shyft.repository.netcdf.arome_data_repository import AromeDataRepository
from shyft import shyftdata_dir
from shapely.geometry import MultiPolygon
from shyft.repository.service.gis_region_model_repository import CellDataFetcher, GridSpecification


class GisRegionModelDemo(object):
    """
    This is just a demo, presenting image plots of the region-models,
    so that we can verify approximate size/shape and content.
    (as well as correct epsg and coordindates)
    """

    def add_plot_polygons(self, ax, polygons, color):
        ps = []
        if not isinstance(polygons, list):
            polygons = [polygons]
        for polygon in polygons:
            if isinstance(polygon, MultiPolygon):
                ps.extend([p for p in polygon])
            else:
                ps.append(polygon)
        patches = map(PolygonPatch, [p for p in ps])
        ax.add_collection(PatchCollection(patches, facecolors=(color,), linewidths=0.1, alpha=0.3))

    def plot_region_model(self, catchment_type, identifier, x0, y0, dx, dy, nx, ny,
                          catch_indicies, station_ids, epsg_id):
        grid_spec = GridSpecification(epsg_id, x0, y0, dx, dy, nx, ny)
        cf = CellDataFetcher(catchment_type, identifier, grid_spec, id_list=catch_indicies)
        print("Start fetching data")
        cf.fetch()
        print("Done, now preparing plot")
        # Plot the extracted data
        fig, ax = plt.subplots(1)
        color_map = {"forest": 'g', "lake": 'b', "glacier": 'r', "cell": "0.75", "reservoir": "purple"}

        extent = grid_spec.geometry[0], grid_spec.geometry[2], grid_spec.geometry[1], grid_spec.geometry[3]
        ax.imshow(cf.elevation_raster, origin='upper', extent=extent, cmap=cm.gray)

        for catchment_cells in iter(cf.cell_data.values()):
            self.add_plot_polygons(ax, [cell["cell"] for cell in catchment_cells], color=color_map["cell"])
        for catchment_land_type in iter(cf.catchment_land_types.values()):
            for k, v in iter(catchment_land_type.items()):
                self.add_plot_polygons(ax, v, color=color_map[k])

        geometry = grid_spec.geometry
        if station_ids is not None:
            glr = GisLocationService()
            stations = glr.get_locations(station_ids, epsg_id)
            if stations:
                points = stations.values()
                ax.scatter([pt[0] for pt in points], [pt[1] for pt in points], alpha=0.5)
                station_min_x = min([v[0] for v in points]) - 1000
                station_max_x = max([v[0] for v in points]) + 1000
                station_min_y = min([v[1] for v in points]) - 1000
                station_max_y = max([v[1] for v in points]) + 1000
                geometry[0] = min(station_min_x, geometry[0])
                geometry[1] = min(station_min_y, geometry[1])
                geometry[2] = max(station_max_x, geometry[2])
                geometry[3] = max(station_max_y, geometry[3])
                base_dir = path.join(shyftdata_dir, "repository", "arome_data_repository")
                EPSG = grid_spec.epsg()
                bbox = grid_spec.bounding_box(EPSG)
                arome4 = AromeDataRepository(EPSG, base_dir, filename="arome_metcoop_default2_5km_*.nc",
                                             bounding_box=bbox, allow_subset=True)
                # data_names = ("temperature", "wind_speed", "precipitation", "relative_humidity","radiation")
                # utc_period = api.UtcPeriod(
                #    api.Calendar().time(2015, 10, 1, 0, 0, 0),
                #    api.Calendar().time(2015, 10, 2, 0, 0, 0)
                # )
                # arome_ts=arome4.get_timeseries(["temperature"],utc_period)
                # arome_points=[gts.mid_point() for gts in arome_ts['temperature']]
                # ax.scatter( [pt.x for pt in arome_points ],[pt.y for pt in arome_points],c=[pt.z for pt in arome_points],alpha=0.5,cmap='gray',s=100)#, facecolors=('r'))

        ax.set_xlim(geometry[0], geometry[2])
        ax.set_ylim(geometry[1], geometry[3])
        plt.show()

    def nea_nidelv(self, identifier):
        x0 = 270000.0
        y0 = 6960000.0
        dx = 1000
        dy = 1000
        nx = 105
        ny = 75
        # Finnkoisjøen, Hersjøen, Lødølja, Nesjøen, Nidarvoll, Sakristian, Selbu II, Sellisjøen, Stuggusjøen, Sylsjøen II, Sørungen, Trondheim-Voll
        #s_list = [121, 217, 360, 421, 423, 489, 503, 506, 574, 598, 610, 632] # Old OBJECTID
        # Finnkoisjøen, Hersjøen, Lødølja, Nesjøen, Sakristian, Selbu II, Sellisjøen, Stuggusjøen, Sylsjøen II, Sørungen, Trondheim-Voll
        s_list = [666, 3, 68, 605, 479, 454, 489, 460, 402, 538, 555] # Updated OBJECTID
        if identifier == 'CATCH_ID':
            # test fetching for regulated catchments using CATCH_ID
            id_list = [1228, 1308, 1330, 1394, 1443, 1726, 1867, 1966, 1996, 2041, 2129, 2195, 2198, 2277, 2402, 2446, 2465, 2545, 2640, 2728, 2718, 3002, 3178, 3536, 3630, 100010, 100011]
        if identifier == 'SUBCATCH_ID':
            # test fetching for regulated catchments using SUBCATCH_ID
            id_list = [188, 196, 180, 191, 187, 190, 172, 177, 174, 185, 175, 176, 183, 179, 197, 171, 181, 182, 184, 186, 189, 192, 194, 195, 752, 173, 195, 193]
        self.plot_region_model('regulated', identifier, x0, y0, dx, dy, nx, ny, id_list, s_list, 32633)

    def vinjevatn(self):
        x0 = 73000.0
        y0 = 6613000.0
        dx = 1000
        dy = 1000
        nx = 40
        ny = 35
        id_list = [446]
        s_list = []
        self.plot_region_model('regulated', 'POWER_PLANT_ID', x0, y0, dx, dy, nx, ny, id_list, s_list, 32633)

    def tistel_33(self):
        id_list = [1225]
        s_list = [619, 684, 129, 654, 218, 542, 650]
        self.plot_region_model('unregulated', 'FELTNR',
                               35000.0, 6788000,
                               1000.0, 1000.0,
                               16, 17, id_list, s_list, 32633)

    def tistel_arome(self):
        id_list = [1225]
        s_list = [619, 684, 129, 654, 218, 542, 650]
        self.plot_region_model('unregulated', 'FELTNR',
                               -10000.0, 6608000,
                               1000.0, 1000.0,
                               200, 230, id_list, s_list, 32633)

    def tistel_32(self):
        id_list = [1225]
        s_list = [619, 684, 129, 654, 218, 542, 650]
        self.plot_region_model('unregulated', 'FELTNR',
                               360000.0, 6761000,
                               1000.0, 1000.0,
                               16, 17, id_list, s_list, 32632)


if __name__ == '__main__':
    demo = GisRegionModelDemo()
    # demo.tistel_32()
    # demo.tistel_arome()
    demo.nea_nidelv('CATCH_ID')
    # demo.vinjevatn()
