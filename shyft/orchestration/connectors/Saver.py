import shutil, os

class save_to_server:
    def __init__ (self,shyftrc,operation,endtime,starttime):
        self.shyftrc = shyftrc
        self.operation = operation
        self.endtime=endtime
        self.starttime = starttime

    def save_state(self,rs,shyftStatePath):
        shyftRunConfig = self.shyftrc
        operation = self.operation
        endtime = self.endtime
        starttime = self.starttime
        state_name      = str.format("{0}.state.{1}-{2:02d}-{3:02d}_{4:02d}.stx", shyftRunConfig.rc.regionName,endtime.year, endtime.month, endtime.day, endtime.hour)
        
        
                
        # The forecasted state would probably only be needed to know the expected snow storage... Whould that information be used? If yes delete 'if' condition
        
        if operation=='Update':        
            
            if os.path.exists(os.path.join(shyftStatePath,'State',state_name)):
                os.remove(os.path.join(shyftStatePath,'State',state_name))
            rs.save_statefile(os.path.join(shyftStatePath,state_name))

    def save_input_database(self,shyftModelPath):
        shyftRunConfig = self.shyftrc
        operation = self.operation
        endtime = self.endtime
        starttime = self.starttime

        input_name      = str.format("{0}.input.{1}-{2:02d}-{3:02d}_{4:02d}.nc", shyftRunConfig.rc.modelName, starttime.year, starttime.month, starttime.day, starttime.hour)
        
        if operation=='Update':
            inp_updt_path   = os.path.join(shyftModelPath,'Input','Update',input_name)
            if os.path.exists(inp_updt_path):
                os.remove(inp_updt_path)
            shutil.move(os.path.join(shyftModelPath,input_name),os.path.join(shyftModelPath,'Input','Update'))

        if operation=='Forecast':
            inp_forec_path = os.path.join(shyftModelPath,'Input','Forecast',input_name)
            if os.path.exists(inp_forec_path):
                os.remove(inp_forec_path)
            shutil.move(os.path.join(shyftModelPath,input_name),os.path.join(shyftModelPath,'Input','Forecast')) 

    def save_output_database(self,shyftModelPath):
        shyftRunConfig = self.shyftrc
        operation = self.operation
        endtime = self.endtime
        starttime = self.starttime
        output_name     = str.format("{0}.output.{1}-{2:02d}-{3:02d}_{4:02d}.nc", shyftRunConfig.rc.modelName,starttime.year, starttime.month, starttime.day, starttime.hour) 
        
        if operation=='Update':   
            out_updt_path = os.path.join(shyftModelPath,'Output','Update',output_name)
            if os.path.exists(out_updt_path):
                os.remove(out_updt_path)
            shutil.move(os.path.join(shyftModelPath,output_name),os.path.join(shyftModelPath,'Output','Update'))

        if operation=='Forecast':
            out_forec_path = os.path.join(shyftModelPath,'Output','Forecast',output_name)
            if os.path.exists(out_forec_path):
                os.remove(out_forec_path)
            shutil.move(os.path.join(shyftModelPath,output_name),os.path.join(shyftModelPath,'Output','Forecast'))