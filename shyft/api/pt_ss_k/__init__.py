from ._pt_ss_k import *
from .. import ByteVector
# Fix up types that we need attached to the model
PTSSKStateVector.push_back = lambda self, x: self.append(x)
PTSSKStateVector.size = lambda self: len(self)


PTSSKModel.cell_t = PTSSKCellAll
PTSSKParameter.map_t = PTSSKParameterMap
PTSSKModel.parameter_t = PTSSKParameter
PTSSKModel.state_t = PTSSKState
PTSSKModel.state_with_id_t = PTSSKStateWithId
PTSSKModel.state = property(lambda self:PTSSKCellAllStateHandler(self.get_cells()))
PTSSKModel.statistics = property(lambda self: PTSSKCellAllStatistics(self.get_cells()))

PTSSKModel.skaugen_snow_state = property(lambda self: PTSSKCellSkaugenStateStatistics(self.get_cells()))
PTSSKModel.skaugen_snow_response = property(lambda self: PTSSKCellSkaugenResponseStatistics(self.get_cells()))
PTSSKModel.priestley_taylor_response = property(lambda self: PTSSKCellPriestleyTaylorResponseStatistics(self.get_cells()))
PTSSKModel.actual_evaptranspiration_response=property(lambda self: PTSSKCellActualEvapotranspirationResponseStatistics(self.get_cells()))
PTSSKModel.kirchner_state = property(lambda self: PTSSKCellKirchnerStateStatistics(self.get_cells()))

PTSSKOptModel.cell_t = PTSSKCellOpt
PTSSKOptModel.parameter_t = PTSSKParameter
PTSSKOptModel.state_t = PTSSKState
PTSSKOptModel.state_with_id_t = PTSSKStateWithId
PTSSKOptModel.state = property(lambda self:PTSSKCellOptStateHandler(self.get_cells()))
PTSSKOptModel.statistics = property(lambda self:PTSSKCellOptStatistics(self.get_cells()))

PTSSKOptModel.optimizer_t = PTSSKOptimizer
PTSSKOptModel.full_model_t =PTSSKModel
PTSSKModel.opt_model_t =PTSSKOptModel
PTSSKModel.create_opt_model_clone = lambda self: create_opt_model_clone(self)
PTSSKModel.create_opt_model_clone.__doc__ = create_opt_model_clone.__doc__
PTSSKOptModel.create_full_model_clone = lambda self: create_full_model_clone(self)
PTSSKOptModel.create_full_model_clone.__doc__ = create_full_model_clone.__doc__

PTSSKCellAll.vector_t = PTSSKCellAllVector
PTSSKCellOpt.vector_t = PTSSKCellOptVector
PTSSKState.vector_t = PTSSKStateVector
#PTSSKState.serializer_t= PTSSKStateIo
PTSSKStateVector.push_back = lambda self, x: self.append(x)

#decorate StateWithId for serialization support
def serialize_to_bytes(state_with_id_vector:PTSSKStateWithIdVector)->ByteVector:
    if not isinstance(state_with_id_vector,PTSSKStateWithIdVector):
        raise RuntimeError("supplied argument must be of type PTSSKStateWithIdVector")
    return serialize(state_with_id_vector)

def __serialize_to_str(state_with_id_vector:PTSSKStateWithIdVector)->str:
    return str(serialize_to_bytes(state_with_id_vector))  # returns hex-string formatted vector

def __deserialize_from_str(s:str)->PTSSKStateWithIdVector:
    return deserialize_from_bytes(ByteVector.from_str(s))

PTSSKStateWithIdVector.serialize_to_bytes = lambda self: serialize_to_bytes(self)
PTSSKStateWithIdVector.serialize_to_str = lambda self: __serialize_to_str(self)
PTSSKStateWithIdVector.deserialize_from_str = __deserialize_from_str
PTSSKStateWithIdVector.state_vector = property(lambda self: extract_state_vector(self),doc=extract_state_vector.__doc__)

def deserialize_from_bytes(bytes: ByteVector)->PTSSKStateWithIdVector:
    if not isinstance(bytes,ByteVector):
        raise RuntimeError("Supplied type must be a ByteVector, as created from serialize_to_bytes")
    states=PTSSKStateWithIdVector()
    deserialize(bytes,states)
    return states