import onnx
from onnx_tf.backend import prepare

# ONNX to TF
onnx_path_fix = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_onnx/checkpoint_epoch_264-new.onnx"
model_onnx = onnx.load(onnx_path_fix)
tf_rep = prepare(model_onnx)

tf_path = "/nfs/masi/kanakap/projects/DeepN4/src/trained_model_tf/checkpoint_epoch_264.pd"
tf_rep.export_graph(tf_path)

