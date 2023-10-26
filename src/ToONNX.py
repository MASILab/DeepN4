import onnx
from onnx import helper
from utils import load_model
from model_all import Synbo_UNet3D

# Load model
model_arch = Synbo_UNet3D(1, 1)
path_to_weights = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_Synbo_UNet3D/checkpoint_epoch_264"
model = load_model(model_arch, path_to_weights)


# PyTorch model 2 ONNX
dummy_input = Variable(torch.randn(1,1, 128, 128, 128))  # Provide a sample input shape
onnx_path = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_onnx/checkpoint_epoch_264.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

# For Tensorflow this workaround was required
onnx_model = onnx.load(onnx_path)

# Define a mapping from old names to new names
name_map = {"input.1": "input_1"}

# Initialize a list to hold the new inputs
new_inputs = []

# Iterate over the inputs and change their names if needed
for inp in onnx_model.graph.input:
    if inp.name in name_map:
        # Create a new ValueInfoProto with the new name
        new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                inp.type.tensor_type.elem_type,
                                                [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
        new_inputs.append(new_inp)
    else:
        new_inputs.append(inp)

# Clear the old inputs and add the new ones
onnx_model.graph.ClearField("input")
onnx_model.graph.input.extend(new_inputs)

# Go through all nodes in the model and replace the old input name with the new one
for node in onnx_model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name in name_map:
            node.input[i] = name_map[input_name]

# Save the renamed ONNX model
onnx_path_fix = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_onnx/checkpoint_epoch_264-new.onnx"
onnx.save(onnx_model, onnx_path_fix)
