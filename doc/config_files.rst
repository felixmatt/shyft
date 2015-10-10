Configuring SHyFT
=================

SHyFT is configured via `YAML files <http://yaml.org/>`_.  Here it is an example::

    ---
    Himalayas:
      config_dir: .     # where to find other config files
      region_config: region.yaml
      model_config: model.yaml
      # model_config: calibrated_model.yaml
      datasets_config: datasets.yaml
      start_datetime: 1990-07-01T00:00:00
      run_time_step: 86400
      number_of_steps: 730
      max_cells: 4  # consider only a maximum of cells (optional, mainly for testing purposes)
      output:
        params:
          - total_discharge
          - discharge
          - snow_storage
          - temperature
          - precipitation
        format: netcdf
        nc_dir: .   # dir where the output file will be stored
        file: output.nc

    ...

[to be continued...]
