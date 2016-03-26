from ._pt_gs_k import *
# Fix up types that we need attached to the model
PTGSKModel.cell_t = PTGSKCellAll
PTGSKParameter.map_t = PTGSKParameterMap
PTGSKModel.parameter_t = PTGSKParameter
PTGSKModel.state_t = PTGSKState
PTGSKModel.statistics = property(lambda self: PTGSKCellAllStatistics(self.get_cells()))
PTGSKModel.gamma_snow_state = property(lambda self: PTGSKCellGammaSnowStateStatistics(self.get_cells()))
PTGSKModel.gamma_snow_response = property(lambda self: PTGSKCellGammaSnowResponseStatistics(self.get_cells()))
#PTGSKModel.priestley_taylor_response = property(lambda self: PTGSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
#PTGSKModel.actual_evaptranspiration_response=property(lambda self: PTGSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
#PTGSKModel.kirchner_state = property(lambda self: PTGSKCellKirchnerStateStatistics(self.get_cells()))
PTGSKOptModel.cell_t = PTGSKCellOpt
PTGSKOptModel.parameter_t = PTGSKParameter
PTGSKOptModel.state_t = PTGSKState
PTGSKOptModel.statistics = property(lambda self:PTGSKCellOptStatistics(self.get_cells()))
#PTGSKOptModel.optimizer_t = PTGSKOptimizer
PTGSKCellAll.vector_t = PTGSKCellAllVector
PTGSKCellOpt.vector_t = PTGSKCellOptVector
PTGSKState.vector_t = PTGSKStateVector
PTGSKState.serializer_t= PTGSKStateIo
