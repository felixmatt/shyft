from __future__ import print_function
import sys,pdb
from os import path 
from RunConfiguration import EnkiRc, EnkiResult
from EnkiCoreAdapter import EnkiCoreAdapter
import enki as enki
import datetime as dt
#import glob
#sys.path.append('D:/Users/jfb/projects/statkraft')
from connectors import smg, Saver
import pyStatkraftScriptAPI.pyStatkraftScriptAPI as Ssa
enkiRootPath = path.abspath(r"D:/enki")

class scheduler:
    def __init__(self,update_step_length,starting_hour,update_length,regions):
        self.update_step_length = update_step_length
        self.starting_hour      = starting_hour
        self.update_length      = update_length
        
        for region in regions:

            self.enkiModelPath = path.join(enkiRootPath, 'Data', region)        
            self.enkiModelJsonPath = path.join(self.enkiModelPath, region+'.json') 
            self.enkiStatePath = path.join(enkiRootPath, 'Data', region,'State')
            print('----------',region,'----------')

            self.define_update_forecast()
    def run_enki(self,options, starttime, endtime):
        '''
        options is a runconfiguration dict for overiding the default parameters from the json file.
        '''


        enkiRunConfig = EnkiRc(self.enkiModelJsonPath, **options)
        enkiRunConfig.initiate_options(self.operation)
        enkiRunConfig.initiate_datasets()
        enkiService = enki.EnkiApiService(False)
        enkiService.setRootPath(self.enkiModelPath)
    
        result = enkiService.run(enkiRunConfig.rc)


        if result.success:
            rs = EnkiResult(result, enkiRunConfig)
            rs.write_results()

        saver = Saver.save_to_server(enkiRunConfig,self.operation,endtime,starttime)
        saver.save_state(rs,self.enkiStatePath)
        saver.save_input_database(self.enkiModelPath)
        saver.save_output_database(self.enkiModelPath)
        # New enki::core run
        eca = EnkiCoreAdapter(enkiRunConfig.rc.regionXml,enkiRunConfig.rc.stateXml,enkiRunConfig.rc.ts)
        eca.test_build_and_run_model(enkiRunConfig.rc.Tstart,enkiRunConfig.rc.deltaT,enkiRunConfig.rc.nSteps)


        return result

    def define_update_forecast(self):

        x = dt.date.today()
        today = dt.datetime(x.year, x.month, x.day, self.starting_hour, 0, 0)

        start = today - dt.timedelta (hours=self.update_length) #initial start of simulation (update period...)
        stop = today + dt.timedelta(hours=240)                 # stop of forecast period
        self.initialize_run_enki(today,start,stop)

    def initialize_run_enki(self,today,start,stop):
        while start < stop :

            if start < today:
                self.operation = 'Update'
                step_length = self.update_step_length
                end = start + dt.timedelta(hours=self.update_step_length)
                if end>today:
                    end = today
            else:
                self.operation = 'Forecast'
                step_length=240
                start = today
                end = stop      
            
        
            start_datetime = start.strftime("%Y-%m-%dT%H:%M:%S")
            options = optionOveride = {'run_configuration': {'number_of_steps': step_length, 
                                                                'run_time_step': '1H',
                                                                'start_datetime' : start_datetime
                                                                }
                                        }

            result = self.run_enki(options, start, end)
            print ('From',start,'to',end)
            print(self.operation,'done. Success:', result.success)
        
            if not result.success:
                 for i in range(result.messageCount()):
                     if result.message(i).strip() != '':
                         print(result.message(i))
                 print("Enki run failed")
                 sys.exit()

            start = end

if __name__ == "__main__":
    
        
    ####----!!!! List all desired regions here !!!!----####
    # the name of the region has to be the same as the name of the folder in D:/enki/Data as well as the name of the json file #
    # 'RanaLang','NeaNidelva',
    regions = ['CentralRegion']

    # options common to all catchments #

    update_step_length  = 24    # how often (hours) to save the state. 24h suggested
    starting_hour       = 0     # when (hour of the day) to start the forecast and end the update 
    update_length       = 240  # how long should the update period be. Robert suggested to extend more than one day (possibly around 3weeks) to include eventual corrections in the input TS.
    scheduler(update_step_length,starting_hour,update_length,regions)

