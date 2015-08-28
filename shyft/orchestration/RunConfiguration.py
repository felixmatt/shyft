# -*- coding: utf-8 -*-

import sys
import yaml
from datetime import datetime

from shyft.orchestration.utils.ShyftRunner import ShyftRunner
from shyft.orchestration.utils.ShyftConfig import ShyftConfig
from shyft import api
import os

#import misc
#import state_update
#import fetch_smg_data


run_modes=['observed']
model_names=['Vik-SHOP-Tistel']
#,
#             'Vik-SHOP-Vik-Hove-lo2014',
#             'tokke-SHOP-Vinjevatn',
#             'rana-SHOP-RanaLangvatn',
#             'nn-SHOP-Stuggusjo'
#             ]
             
obs_series_f = "P:/projects/config_auto/observed_inflow_series.yml"

forecast_period=864000 # seconds in 10days

enki_root=os.path.join("P:/","projects")
config_file = os.path.join(enki_root, "config_auto","runner_configurations_auto.yaml")

# Using python datetime
#current_d = datetime.utcnow()
#log.write(current_d.strftime('%Y%m%dT%H%M')+'\n')
#current_d = current_d.replace(hour=0, minute=0, second=0, microsecond=0)-timedelta(hours=1)
#current_d_timestamp = int(round((current_d - datetime.utcfromtimestamp(0)).total_seconds()))

# Using enki Calendar
cal=api.Calendar()
current_d_timestamp = cal.trim(api.utctime_now(),api.Calendar.DAY)-3600
#current_d = cal.toString(current_d_timestamp)
current_d = datetime.utcfromtimestamp(current_d_timestamp)
#log.write(cal.toString(current_d_timestamp)+'\n')

#log.close()

for model_name in model_names:
    for run_mode in run_modes:

        if(run_mode=='observed'):
            config_name=model_name+'_observed'
            with open(config_file) as f:
                dct = yaml.load(f)
            val=dct[config_name]
            #init_state_timestamp=int(round((val['start_datetime'] - datetime.utcfromtimestamp(0)).total_seconds())) # Python datetime
            d=val['start_datetime']
            init_state_timestamp=cal.time(api.YMDhms(d.year,d.month,d.day,d.hour,d.minute,d.second)) # enki calendar
            n_steps=int((current_d_timestamp-init_state_timestamp)/val['run_time_step'])
            val['number_of_steps']=n_steps
            with open(config_file, "w") as f:
                yaml.dump(dct, f)
                
        if(run_mode=='forecast'):
            config_name=model_name+'_forecast'
            with open(config_file) as f:
                dct = yaml.load(f)
            val=dct[config_name]
            val['start_datetime']=current_d
            n_steps=int(forecast_period/val['run_time_step'])
            val['number_of_steps']=n_steps
            with open(config_file, "w") as f:
                yaml.dump(dct, f)
        
        config= ShyftConfig(config_file,config_name)

        simulator = ShyftRunner(config)
        simulator.build_model(config.t_start, config.dt, config.n_steps)
        simulator.run_model()
        #if(run_mode=='forecast'):
        #    obs_state=fetch_smg_data.get_obs_inflow(model_name,obs_series_f,config.t_start-config.dt, n_steps=24,dt=config.dt)
        #    state_update.update_q(simulator,obs_state)
        #    misc.change_uid(simulator._config)
        #    simulator.run_model()
            
            
