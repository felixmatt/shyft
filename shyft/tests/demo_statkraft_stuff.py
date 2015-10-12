import matplotlib.pylab as plt
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib import cm
from itertools import imap


from shapely.geometry import MultiPolygon #Polygon, MultiPolygon, box, Point

from shyft.repository.service.gis_region_model_repository import CellDataFetcher,GridSpecification

class GisRegionModelDemo(object):
    """
    This is just a demo, presenting image plots of the region-models,
    so that we can verify approximate size/shape and content.
    (as well as correct epsg and coordindates)
    """
    
    def add_plot_polygons(self,ax, polygons, color):
        ps = []
        if not isinstance(polygons, list):
            polygons = [polygons]
        for polygon in polygons:
            if isinstance(polygon, MultiPolygon):
                ps.extend([p for p in polygon])
            else:
                ps.append(polygon)
        patches = imap(PolygonPatch, [p for p in ps])
        ax.add_collection(PatchCollection(patches, facecolors=(color,), linewidths=0.1, alpha=0.3))
    
    def plot_region_model(self,catchment_type,identifier,x0, y0, dx, dy, nx, ny, catch_indicies,epsg_id):
        grid_spec=GridSpecification(epsg_id,x0, y0, dx, dy, nx, ny)
        cf = CellDataFetcher(catchment_type,identifier,grid_spec, id_list=catch_indicies)
        print( "Start fetching data")
        cf.fetch()
        print ("Done, now preparing plot")
        # Plot the extracted data
        fig, ax = plt.subplots(1)
        color_map = {"forest": 'g', "lake": 'b', "glacier": 'r', "cell": "0.75", "reservoir": "purple"}
    
        extent = grid_spec.geometry[0], grid_spec.geometry[2], grid_spec.geometry[1], grid_spec.geometry[3]
        ax.imshow(cf.elevation_raster, origin='upper', extent=extent, cmap=cm.gray)
    
        for catchment_cells in cf.cell_data.itervalues():
            self.add_plot_polygons(ax, [cell["cell"] for cell in catchment_cells], color=color_map["cell"])
        for catchment_land_type in cf.catchment_land_types.itervalues():
            for k,v in catchment_land_type.iteritems():
                self.add_plot_polygons(ax, v, color=color_map[k])
    
        geometry = grid_spec.geometry
        ax.set_xlim(geometry[0], geometry[2])
        ax.set_ylim(geometry[1], geometry[3])
        plt.show()
    
    def nea_nidelv(self):
        x0 = 270000.0
        y0 = 6960000.0
        dx = 1000
        dy = 1000
        nx = 105
        ny = 75
        # test fetching for regulated catchments using CATCH_ID
        id_list=[1228,1308,1394,1443,1726,1867,1996,2041,2129,2195,2198,2277,2402,2446,2465,2545,2640,2718,3002,3536,3630,1000010,1000011]
        self.plot_region_model('regulated','CATCH_ID',x0, y0, dx, dy, nx, ny, id_list,32633)
        
        
    def vinjevatn(self):
        x0 = 73000.0
        y0 = 6613000.0
        dx = 1000
        dy = 1000
        nx = 40
        ny = 35
        id_list=[446]
        self.plot_region_model('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, id_list,32633)
        
        
    def tistel(self):
        x0 = 350000.0
        y0 = 6751000.0
        dx = 1000
        dy = 1000
        nx = 32
        ny = 24
        #id_list = [163,287,332]
        #run_cell_example2('regulated','POWER_PLANT_ID',x0, y0, dx, dy, nx, ny, id_list)
        id_list = [1225,1226]
        self.plot_region_model('unregulated','FELTNR',x0, y0, dx, dy, nx, ny, id_list,32632)

if __name__ == '__main__':
    demo=GisRegionModelDemo()
    demo.tistel()
    demo.nea_nidelv()
    demo.vinjevatn()