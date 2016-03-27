from ._pt_ss_k import *
# Fix up types that we need attached to the model
PTSSKStateVector.push_back = lambda self, x: self.append(x)
PTSSKStateVector.size = lambda self: len(self)


PTSSKModel.cell_t = PTSSKCellAll
PTSSKParameter.map_t = PTSSKParameterMap
PTSSKModel.parameter_t = PTSSKParameter
PTSSKModel.state_t = PTSSKState
PTSSKModel.statistics = property(lambda self: PTSSKCellAllStatistics(self.get_cells()))
PTSSKModel.gamma_snow_state = property(lambda self: PTSSKCellGammaSnowStateStatistics(self.get_cells()))
PTSSKModel.gamma_snow_response = property(lambda self: PTSSKCellGammaSnowResponseStatistics(self.get_cells()))
PTSSKModel.priestley_taylor_response = property(lambda self: PTSSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTSSKModel.actual_evaptranspiration_response=property(lambda self: PTSSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTSSKModel.kirchner_state = property(lambda self: PTSSKCellKirchnerStateStatistics(self.get_cells()))
PTSSKOptModel.cell_t = PTSSKCellOpt
PTSSKOptModel.parameter_t = PTSSKParameter
PTSSKOptModel.state_t = PTSSKState
PTSSKOptModel.statistics = property(lambda self:PTSSKCellOptStatistics(self.get_cells()))
PTSSKOptModel.optimizer_t = PTSSKOptimizer
PTSSKCellAll.vector_t = PTSSKCellAllVector
PTSSKCellOpt.vector_t = PTSSKCellOptVector
PTSSKState.vector_t = PTSSKStateVector
PTSSKState.serializer_t= PTSSKStateIo
PTSSKStateVector.push_back = lambda self, x: self.append(x)
