from station_data import StationDataFetcher
import yaml
import numpy as np

def wash(region_config, dataset_config, max_snapping_distance=500.0):
    max_dist = 1000000000.0 if len(sys.argv) == 1 else float(sys.argv[1])
    xys = []
    old_station_inds = []
    for index in region_config["stations"]["indices"]:
        old_station_inds.append(index["id"])
        xys.append(np.array([index["Xcoord"], index["Ycoord"]]))

    indices = None
    epsg_id = region_config["domain"]["EPSG"]
    sf = StationDataFetcher(indices=indices, epsg_id=epsg_id)
    stations = sf.fetch_data()
    db_inds = []
    db_xys = []
    db_owners = []
    db_names = []
    for i, idx in enumerate(stations):
        db_xy = np.array(stations[idx][0][:2], dtype=np.float64)
        if not np.any(np.isnan(db_xy)):
            db_inds.append(idx)
            db_owners.append(stations[idx][1]["owner"])
            db_names.append(stations[idx][1]["name"])
            db_xys.append(list(db_xy))

    db_xys = np.array(db_xys)
    
    db_xys = db_xys
    diffs= []
    indices = []
    stations_by_uids = {}
    s2uid = {}
    uids = []
    for i, xy in enumerate(xys):
        diff = np.linalg.norm(db_xys - xy, axis=1)
        if min(diff) < max_snapping_distance:
            station = region_config["stations"]["indices"][i]
            diffs.append(min(diff))
            idx = np.linalg.norm(db_xys - xy, axis=1).argmin()
            uid = int(db_inds[idx])
            if uid not in uids: uids.append(uid)
            station["uid"] = uid
            station["name"] = db_names[idx]
            station.pop("Xcoord")
            station.pop("Ycoord")
            indices.append(station)
            if not uid in stations_by_uids: stations_by_uids[uid] = []
            stations_by_uids[uid].append(station)
            s2uid[station["id"]] = uid
        else:
            idx = diff.argmin()
            print "Ignoring: {}, {}, {} dist {}".format(xy, db_names[idx], db_xys[idx], min(diff))
    # Remap old station indices in dataset file and discard stations with unknown locations
    region_config["stations"]["indices"] = indices
    for source in dataset_config["sources"]:
        for tp in source["types"]:
            valid_entries = []
            for station in tp["stations"]:
                station_id = station["station_id"]
                if station_id in s2uid:
                    new_id = uids.index(s2uid[station_id])
                    station["station_id"] = new_id
                    valid_entries.append(station)
            tp["stations"][:] = valid_entries
    # Remap destinations too, although this will be changed in the future to deal with timeseries on a catchment
    # level
    if "destinations" in dataset_config:
        for dest in dataset_config["destinations"]:
            valid_entries = []
            for station in dest["targets"]:
                station_id = station["station_id"]
                if station_id in s2uid:
                    new_id = uids.index(s2uid[station_id])
                    station["station_id"] = new_id
                    valid_entries.append(station)
            dest["targets"][:] = valid_entries

    # Finally, also remap the region config station indices:
    added_inds = []
    unique_stations = []
    for index in region_config["stations"]["indices"]:
        index["id"] = uids.index(s2uid[index["id"]])
        if index["id"] not in added_inds:
            added_inds.append(index["id"])
            unique_stations.append(index)
    region_config["stations"]["indices"][:] = unique_stations 

    return region_config, dataset_config
        

if __name__ == "__main__":
    import sys
    import os
    r_file = r"D:/Users/os/projects/shyft/doc/example/NeaNidelva/region.yaml"
    d_file = r"D:/Users/os/projects/shyft/doc/example/NeaNidelva/datasets.yaml"
    with open(r_file, "r") as ff:
        config = yaml.load(ff.read())
    with open(d_file, "r") as ff:
        dataset_config = yaml.load(ff.read())
    max_snapping_distance = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0
    config, dataset_config = wash(config, dataset_config, max_snapping_distance=max_snapping_distance)
    ro_file = "_out".join(os.path.splitext(r_file))
    do_file = "_out".join(os.path.splitext(d_file))
    with open(ro_file, "w") as of:
        of.write(yaml.dump(config, default_flow_style=False))
    with open(do_file, "w") as of:
        of.write(yaml.dump(dataset_config, default_flow_style=False))

