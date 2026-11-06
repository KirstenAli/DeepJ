"""Finite-difference gradients from ONNX Runtime, without DeepJ arithmetic."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/fixtures'
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
for name, parameters in [('mlp', ['W', 'B']), ('conv2d', ['W']), ('gru_forward', ['W', 'R']), ('transformer', ['WQ', 'WV'])]:
    model = onnx.load(ROOT / (name + '.onnx'))
    document = json.loads((ROOT / (name + '.json')).read_text())
    feed = {key: np.array(v['data'], dtype={1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_}[v['type']]).reshape(v['shape']) for key, v in document['inputs'].items()}
    weights = {t.name: nh.to_array(t).copy() for t in model.graph.initializer}
    for key in parameters:
        model.graph.input.append(h.make_tensor_value_info(key, onnx.TensorProto.FLOAT, weights[key].shape))
        feed[key] = weights[key]
    retained = [t for t in model.graph.initializer if t.name not in parameters]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    def loss():
        output = session.run(['Y'], feed)[0].astype(np.float64)
        return float(np.sum(output * output))
    gradients = {}
    for key in ['X'] + parameters:
        array = feed[key]
        derivative = np.empty_like(array)
        for coordinate in np.ndindex(array.shape):
            original = array[coordinate].copy()
            array[coordinate] = original + np.float32(.001)
            plus = loss()
            array[coordinate] = original - np.float32(.001)
            minus = loss()
            array[coordinate] = original
            derivative[coordinate] = (plus - minus) / .002
        gradients[key] = dict(shape=list(array.shape), data=derivative.reshape(-1).tolist(), type=1)
    (ROOT / (name + '.grad.json')).write_text(json.dumps(dict(parameters=parameters, gradients=gradients), indent=2) + '\n')
    print(name)
