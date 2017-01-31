from ._pt_gs_k import *
from .. import ByteVector
# Fix up types that we need attached to the model
PTGSKStateVector.push_back = lambda self, x: self.append(x)
PTGSKStateVector.size = lambda self: len(self)


PTGSKModel.cell_t = PTGSKCellAll
PTGSKParameter.map_t = PTGSKParameterMap
PTGSKModel.parameter_t = PTGSKParameter
PTGSKModel.state_t = PTGSKState
PTGSKModel.state_with_id_t = PTGSKStateWithId
PTGSKModel.state = property(lambda self:PTGSKCellAllStateHandler(self.get_cells()))
PTGSKModel.statistics = property(lambda self: PTGSKCellAllStatistics(self.get_cells()))

PTGSKModel.gamma_snow_state = property(lambda self: PTGSKCellGammaSnowStateStatistics(self.get_cells()))
PTGSKModel.gamma_snow_response = property(lambda self: PTGSKCellGammaSnowResponseStatistics(self.get_cells()))
PTGSKModel.priestley_taylor_response = property(lambda self: PTGSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTGSKModel.actual_evaptranspiration_response=property(lambda self: PTGSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTGSKModel.kirchner_state = property(lambda self: PTGSKCellKirchnerStateStatistics(self.get_cells()))

PTGSKOptModel.cell_t = PTGSKCellOpt
PTGSKOptModel.parameter_t = PTGSKParameter
PTGSKOptModel.state_t = PTGSKState
PTGSKOptModel.state_with_id_t = PTGSKStateWithId
PTGSKOptModel.state = property(lambda self:PTGSKCellOptStateHandler(self.get_cells()))
PTGSKOptModel.statistics = property(lambda self:PTGSKCellOptStatistics(self.get_cells()))

PTGSKOptModel.optimizer_t = PTGSKOptimizer
PTGSKOptModel.full_model_t =PTGSKModel
PTGSKModel.opt_model_t =PTGSKOptModel


PTGSKCellAll.vector_t = PTGSKCellAllVector
PTGSKCellOpt.vector_t = PTGSKCellOptVector
PTGSKState.vector_t = PTGSKStateVector
PTGSKState.serializer_t= PTGSKStateIo

#decorate StateWithId for serialization support
def serialize_to_bytes(state_with_id_vector):
    if not isinstance(state_with_id_vector,PTGSKStateWithIdVector):
        raise RuntimeError("supplied argument must be of type PTGSKStateWithIdVector")
    return serialize(state_with_id_vector)

PTGSKStateWithIdVector.serialize_to_bytes = lambda self: serialize_to_bytes(self)

def deserialize_from_bytes(bytes):
    if not isinstance(bytes,ByteVector):
        raise RuntimeError("Supplied type must be a ByteVector, as created from serialize_to_bytes")
    states=PTGSKStateWithIdVector()
    deserialize(bytes,states)
    return states