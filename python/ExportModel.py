# %%
from python.ModelTraining import model
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 3]))]
onx = to_onnx(model, initial_type)
# %%
