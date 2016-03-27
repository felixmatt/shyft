from ._pt_hs_k import *
# Fix up types that we need attached to the model
PTHSKStateVector.push_back = lambda self, x: self.append(x)
PTHSKStateVector.size = lambda self: len(self)

PTHSKModel.cell_t = PTHSKCellAll
PTHSKParameter.map_t = PTHSKParameterMap
PTHSKModel.parameter_t = PTHSKParameter
PTHSKModel.state_t = PTHSKState
PTHSKModel.statistics = property(lambda self: PTHSKCellAllStatistics(self.get_cells()))
PTHSKModel.gamma_snow_state = property(lambda self: PTHSKCellGammaSnowStateStatistics(self.get_cells()))
PTHSKModel.gamma_snow_response = property(lambda self: PTHSKCellGammaSnowResponseStatistics(self.get_cells()))
PTHSKModel.priestley_taylor_response = property(lambda self: PTHSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTHSKModel.actual_evaptranspiration_response=property(lambda self: PTHSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTHSKModel.kirchner_state = property(lambda self: PTHSKCellKirchnerStateStatistics(self.get_cells()))
PTHSKOptModel.cell_t = PTHSKCellOpt
PTHSKOptModel.parameter_t = PTHSKParameter
PTHSKOptModel.state_t = PTHSKState
PTHSKOptModel.statistics = property(lambda self:PTHSKCellOptStatistics(self.get_cells()))
PTHSKOptModel.optimizer_t = PTHSKOptimizer
PTHSKCellAll.vector_t = PTHSKCellAllVector
PTHSKCellOpt.vector_t = PTHSKCellOptVector
PTHSKState.vector_t = PTHSKStateVector
PTHSKState.serializer_t= PTHSKStateIo

