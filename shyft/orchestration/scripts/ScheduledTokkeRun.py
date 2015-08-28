from __future__ import print_function
import sys 
import os
from os import path 
from RunConfiguration import EnkiRc, EnkiResult
import EnkiApiPyModule as enki
import pdb
import datetime as dt
import shutil
import glob

def run_enki(options, starttime, endtime):
    '''
    options is a runconfiguration dict for overiding the default parameters from the json file.
    '''
    # path management
    enkiRootPath = path.abspath(r"D:/enki")
    enkiBinPath = path.join(enkiRootPath, 'bin')
    enkiModelPath = path.join(enkiRootPath, 'Data', 'Tokke_Prod')
    enkiModelJsonPath = path.join(enkiModelPath, 'Tokke_Prod_fullnetwork.json')  # 'Tokke_Prod.json'

    enkiRunConfig = EnkiRc(enkiModelJsonPath, **options)
    enkiRunConfig.initiate_options()
    enkiRunConfig.initiate_datasets()
    enkiService = enki.EnkiApiService(False)
    enkiService.setRootPath(enkiModelPath)
    
    result = enkiService.run(enkiRunConfig.rc)
    stateFile = os.path.join(enkiModelPath, str.format("{0}.state.{1}-{2:02d}-{3:02d}_{4:02d}.stx", 'ENKI', 
                                                       endtime.year, endtime.month, endtime.day, endtime.hour))
    if result.success:
        rs = EnkiResult(result, enkiRunConfig)
        rs.write_results()
        
        rs.save_statefile(stateFile)
        
    return result


if __name__ == "__main__":

    #today = dt.datetime.now()
    #end = dt.timedelta(hoursdt.datetime(2014, 3, 3, 8, 0, 0)
    today = dt.datetime(2013, 6, 13, 8, 0, 0)
    start = today
    end = dt.datetime.now()
    while start < end:
    
        start_datetime = start.strftime("%Y-%m-%dT%H:%M:%S")
        options = optionOveride = {'run_configuration': {'number_of_steps': 240, 
                                                        'run_time_step': '1H',
                                                        'start_datetime' : start_datetime
                                                        }
                                }

        endtime = start + dt.timedelta(hours=240)
        result = run_enki(options, start, endtime)
        print("For Date:", start)
        print("Done. Success: ", result.success)
        
        if not result.success:
            for i in range(result.messageCount()):
                if result.message(i).strip() != '':
                    print(result.message(i))
            print("Enki run failed")
            sys.exit()

        
        #self.stateXmlFile = str.format("{0}.state.{1}-{2:02d}-{3:02d}_{4:02d}.stx", self.rc.modelName, start.year, start.month, start.day, start.hour)
        
        start = endtime

                          
    enkiRootPath = path.abspath(r"D:/enki")
    enkiModelPath = path.join(enkiRootPath, 'Data', 'Tokke_Prod')
    for f in glob.glob(path.join(enkiModelPath, 'ENKI.[output|input]*')):
        basename, fname = path.split(f)
        if path.exists(path.join(basename, 'archive', fname)):
            os.unlink(path.join(basename, 'archive', fname))

        shutil.move(f, path.join(enkiModelPath, 'archive'))