from ._pt_hs_k import *
# Fix up types that we need attached to the model
PTHSKStateVector.push_back = lambda self, x: self.append(x)
PTHSKStateVector.size = lambda self: len(self)

PTHSKModel.cell_t = PTHSKCellAll
PTHSKParameter.map_t = PTHSKParameterMap
PTHSKModel.parameter_t = PTHSKParameter
PTHSKModel.state_t = PTHSKState
PTHSKModel.state_with_id_t = PTHSKStateWithId
PTHSKModel.state = property(lambda self:PTHSKCellAllStateHandler(self.get_cells()))
PTHSKModel.statistics = property(lambda self: PTHSKCellAllStatistics(self.get_cells()))

PTHSKModel.hbv_snow_state = property(lambda self: PTHSKCellHBVSnowStateStatistics(self.get_cells()))
PTHSKModel.hbv_snow_response = property(lambda self: PTHSKCellHBVSnowResponseStatistics(self.get_cells()))
PTHSKModel.priestley_taylor_response = property(lambda self: PTHSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTHSKModel.actual_evaptranspiration_response=property(lambda self: PTHSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTHSKModel.kirchner_state = property(lambda self: PTHSKCellKirchnerStateStatistics(self.get_cells()))

PTHSKOptModel.cell_t = PTHSKCellOpt
PTHSKOptModel.parameter_t = PTHSKParameter
PTHSKOptModel.state_t = PTHSKState
PTHSKOptModel.state_with_id_t = PTHSKStateWithId
PTHSKOptModel.state = property(lambda self:PTHSKCellOptStateHandler(self.get_cells()))
PTHSKOptModel.statistics = property(lambda self:PTHSKCellOptStatistics(self.get_cells()))

PTHSKOptModel.optimizer_t = PTHSKOptimizer

PTHSKOptModel.full_model_t =PTHSKModel
PTHSKModel.opt_model_t =PTHSKOptModel
PTHSKModel.create_opt_model_clone = lambda self: create_opt_model_clone(self)
PTHSKModel.create_opt_model_clone.__doc__ = create_opt_model_clone.__doc__
PTHSKOptModel.create_full_model_clone = lambda self: create_full_model_clone(self)
PTHSKOptModel.create_full_model_clone.__doc__ = create_full_model_clone.__doc__

PTHSKCellAll.vector_t = PTHSKCellAllVector
PTHSKCellOpt.vector_t = PTHSKCellOptVector
PTHSKState.vector_t = PTHSKStateVector
PTHSKState.serializer_t= PTHSKStateIo

#decorate StateWithId for serialization support
def serialize_to_bytes(state_with_id_vector):
    if not isinstance(state_with_id_vector,PTHSKStateWithIdVector):
        raise RuntimeError("supplied argument must be of type PTHSKStateWithIdVector")
    return serialize(state_with_id_vector)

PTHSKStateWithIdVector.serialize_to_bytes = lambda self: serialize_to_bytes(self)

def deserialize_from_bytes(bytes):
    if not isinstance(bytes,ByteVector):
        raise RuntimeError("Supplied type must be a ByteVector, as created from serialize_to_bytes")
    states=PTHSKStateWithIdVector()
    deserialize(bytes,states)
    return states