#pragma once

/** \brief CoreApiCellBuiliderTest verifies how to assemble/construct Cells from geo-input specification
 * Typical use-case:
 *  Have
 *     1. a  region grid.spec (x0,y0,delta, nx,ny) --> nx * ny boxes with box ( x0+i*delta,y0+i*delta,delta,delta)
 *     2. catchment boundaries:polygons
 *     3. terrain properties, like lake,forest,glacier,reservoirs: polygons
 *     3. digital terrain model:
 *             z= dtm.z_avg(box) or dtm.z(x,y)
 *             slope=dtm.slope(box)
 *             aspect=dtm.aspect(box)
 *         maybe
 *             radiation_factor=dtm.radiation_factor(box)
 *  output:
 *     Collection of enki GeoCellData
 */

class cell_builder_test : public CxxTest::TestSuite {
  public:
    void test_read_geo_region_data_from_files(void);
    void test_io_performance(void);
    void test_read_geo_point_map(void);
    void test_read_geo_located_ts(void);
    void test_read_and_run_region_model(void);

};
