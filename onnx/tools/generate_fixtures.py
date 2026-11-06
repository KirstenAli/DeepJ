"""Small independent interoperability oracles. Requires onnx 1.23.2 + onnxruntime 1.31.1."""
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/fixtures'
RNG = np.random.default_rng(7351)
MANIFEST = []
OPTIONS = ort.SessionOptions()
OPTIONS.intra_op_num_threads = 1
OPTIONS.inter_op_num_threads = 1


def rand(*shape):
    return RNG.normal(0, .3, shape).astype(np.float32)


def value(name, array):
    return h.make_tensor_value_info(name, nh.from_array(array).data_type, array.shape)


def describe(array):
    array = np.asarray(array)
    return dict(type=nh.from_array(array).data_type, shape=list(array.shape), data=array.reshape(-1).tolist())


def case(name, nodes, inputs, weights=None, outputs=('Y',), opset=18, functions=(), reference=False):
    weights = weights or {}
    graph = h.make_graph(nodes, name, [value(k, v) for k, v in inputs.items()],
                         [h.make_value_info(n, onnx.TypeProto()) for n in outputs],
                         [nh.from_array(v, k) for k, v in weights.items()])
    imports = [h.make_opsetid('', opset)] + [h.make_opsetid(f.domain, 1) for f in functions]
    model = h.make_model(graph, opset_imports=imports, functions=functions, ir_version=10)
    # Reference shape/type inference supplies exact integer/boolean output declarations.
    model = onnx.shape_inference.infer_shapes(model)
    if reference:
        result = ReferenceEvaluator(model).run(list(outputs), inputs)
        engine = 'ONNX ReferenceEvaluator 1.23.2'
    else:
        result = ort.InferenceSession(model.SerializeToString(), OPTIONS, providers=['CPUExecutionProvider']).run(list(outputs), inputs)
        engine = 'ONNX Runtime 1.31.1 CPU'
    for decl, array in zip(model.graph.output, result):
        decl.CopyFrom(value(decl.name, np.asarray(array)))
    onnx.checker.check_model(model)
    onnx.save_model(model, ROOT / (name + '.onnx'))
    document = dict(engine=engine, inputs={k: describe(v) for k, v in inputs.items()},
                    outputs={k: describe(v) for k, v in zip(outputs, result)})
    (ROOT / (name + '.json')).write_text(json.dumps(document, indent=2) + '\n')
    MANIFEST.append(name)
    print(name)
    return model


def node(op, inputs, outputs=('Y',), **attributes):
    return h.make_node(op, inputs, list(outputs), **attributes)


case('mlp', [node('Gemm', ['X', 'W', 'B'], ['A']), node('Relu', ['A'])],
     {'X': rand(2, 3)}, {'W': rand(3, 4), 'B': rand(4)})
for op in ['Add', 'Sub', 'Mul', 'Div', 'MatMul']:
    a, b = (rand(2, 3), rand(3, 4)) if op == 'MatMul' else (rand(2, 3), rand(3) + 1)
    case(op.lower(), [node(op, ['X', 'B'])], {'X': a}, {'B': b})
for op in ['Abs', 'Neg', 'Exp', 'Log', 'Sqrt', 'Reciprocal', 'Relu', 'Sigmoid', 'Tanh',
           'Softplus', 'Softsign', 'Erf', 'LeakyRelu', 'Elu', 'Selu', 'HardSigmoid', 'HardSwish', 'Mish']:
    x = np.array([-2, -.1, 0, .1, 2], np.float32)
    if op in ['Log', 'Sqrt', 'Reciprocal']:
        x += 3
    case(op.lower(), [node(op, ['X'])], {'X': x})
for approx in ['none', 'tanh']:
    case('gelu_' + approx, [node('Gelu', ['X'], approximate=approx)], {'X': rand(2, 3)}, opset=20)
case('pow', [node('Pow', ['X', 'P'])], {'X': np.array([-2, 0, 3], np.float32)}, {'P': np.array(3, np.float32)})
case('prelu', [node('PRelu', ['X', 'B'])], {'X': rand(2, 3)}, {'B': np.array([.1, .2, .3], np.float32)})
case('clip', [node('Clip', ['X', 'A', 'B'])], {'X': rand(2, 3)}, {'A': np.array(-.2, np.float32), 'B': np.array(.3, np.float32)})
for op in ['Sum', 'Mean', 'Max', 'Min']:
    case('variadic_' + op.lower(), [node(op, ['X', 'A', 'B'])], {'X': rand(2, 3)}, {'A': rand(3), 'B': rand(2, 1)})
for op in ['Equal', 'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual']:
    case(op.lower(), [node(op, ['X', 'B'])], {'X': rand(2, 3)}, {'B': rand(3)})
for op in ['And', 'Or', 'Xor']:
    case(op.lower(), [node(op, ['X', 'B'])], {'X': np.array([[True], [False]])}, {'B': np.array([True, False, True])})
case('not', [node('Not', ['X'])], {'X': np.array([True, False])})
case('where', [node('Where', ['C', 'X', 'B'])], {'X': rand(2, 3)}, {'C': np.array([[True], [False]]), 'B': rand(3)})
case('integer_exact', [node('Add', ['X', 'B'])], {'X': np.array([2**40 + 1, -2**40], np.int64)}, {'B': np.array(3, np.int64)})
case('shape_dynamic', [node('Shape', ['X'], ['S']), node('Size', ['X'], ['Z']), node('Flatten', ['X'], ['F'])],
     {'X': rand(2, 3, 4)}, outputs=['S', 'Z', 'F'])
case('reshape_transpose', [node('Reshape', ['X', 'S'], ['A']), node('Transpose', ['A'], perm=[1, 0])],
     {'X': rand(2, 3)}, {'S': np.array([3, -1], np.int64)})
case('expand_tile', [node('Expand', ['X', 'S'], ['A']), node('Tile', ['A', 'R'])],
     {'X': rand(1, 3)}, {'S': np.array([2, 3], np.int64), 'R': np.array([1, 2], np.int64)})
case('squeeze_split', [node('Unsqueeze', ['X', 'A'], ['U']), node('Squeeze', ['U', 'A'], ['S']),
                       node('Split', ['S', 'P'], ['P0', 'P1'], axis=1), node('Concat', ['P1', 'P0'], axis=1)],
     {'X': rand(2, 3)}, {'A': np.array([0, 3], np.int64), 'P': np.array([1, 2], np.int64)})
case('slice_reverse', [node('Slice', ['X', 'S', 'E', 'A', 'P'])], {'X': rand(2, 3, 4)},
     {k: np.array(v, np.int64) for k, v in dict(S=[-1, 0], E=[np.iinfo(np.int64).min, 3], A=[2, 1], P=[-1, 2]).items()})
case('gather', [node('Gather', ['X', 'I'], axis=1)], {'X': rand(2, 3, 4)}, {'I': np.array([[2, -1], [0, 1]], np.int64)})
case('gather_elements', [node('GatherElements', ['X', 'I'], axis=1)], {'X': rand(2, 3)}, {'I': np.array([[2, 0], [1, -1]], np.int64)})
case('gather_nd', [node('GatherND', ['X', 'I'], batch_dims=1)], {'X': rand(2, 3, 4)}, {'I': np.array([[[2], [0]], [[1], [-1]]], np.int64)})
case('constant_shape', [node('Constant', [], ['V'], value=nh.from_array(np.array([2, 3], np.int64))),
                        node('ConstantOfShape', ['V'], value=nh.from_array(np.array([.25], np.float32)))], {})
case('cast', [node('Cast', ['X'], ['A'], to=T.INT64), node('CastLike', ['A', 'X'])], {'X': np.array([-3.5, .5, 2.9], np.float32)})
case('range', [node('Range', ['A', 'B', 'C'])], {}, {'A': np.array(1, np.int64), 'B': np.array(8, np.int64), 'C': np.array(2, np.int64)})
for op in ['ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceL1', 'ReduceL2',
           'ReduceLogSum', 'ReduceLogSumExp', 'ReduceSumSquare']:
    case(op.lower(), [node(op, ['X', 'A'], keepdims=0)], {'X': np.abs(rand(2, 3, 4)) + .2}, {'A': np.array([0, 2], np.int64)})
for op in ['Softmax', 'LogSoftmax']:
    case(op.lower(), [node(op, ['X'], axis=1)], {'X': rand(2, 3, 4)})
    case(op.lower() + '_legacy', [node(op, ['X'], axis=1)], {'X': rand(2, 3, 4)}, opset=11)
for op in ['ArgMax', 'ArgMin']:
    case(op.lower(), [node(op, ['X'], axis=1, keepdims=0, select_last_index=1)], {'X': np.array([[2, 2, 1], [3, 0, 3]], np.float32)})
case('topk', [node('TopK', ['X', 'K'], ['Y', 'I'], axis=-1)], {'X': rand(2, 4)}, {'K': np.array([2], np.int64)}, outputs=['Y', 'I'])
case('layer_norm', [node('LayerNormalization', ['X', 'S', 'B'], ['Y', 'M', 'V'], axis=-2)],
     {'X': rand(2, 3, 4)}, {'S': rand(3, 4), 'B': rand(3, 4)}, outputs=['Y', 'M', 'V'])
case('instance_norm', [node('InstanceNormalization', ['X', 'S', 'B'])], {'X': rand(2, 4, 3, 3)}, {'S': rand(4), 'B': rand(4)})
case('batch_norm', [node('BatchNormalization', ['X', 'S', 'B', 'M', 'V'])], {'X': rand(2, 4, 3, 3)},
     {'S': rand(4), 'B': rand(4), 'M': rand(4), 'V': np.abs(rand(4)) + .5})
case('batch_norm_training', [node('BatchNormalization', ['X', 'S', 'B', 'M', 'V'], ['Y', 'R', 'Q'], training_mode=1)],
     {'X': rand(2, 4, 3, 3)}, {'S': rand(4), 'B': rand(4), 'M': rand(4), 'V': np.abs(rand(4)) + .5}, outputs=['Y', 'R', 'Q'])
case('group_norm', [node('GroupNormalization', ['X', 'S', 'B'], num_groups=2)], {'X': rand(2, 4, 3, 3)}, {'S': rand(4), 'B': rand(4)}, opset=21)
case('dropout', [node('Dropout', ['X'], ['Y', 'M'])], {'X': rand(2, 3)}, outputs=['Y', 'M'])
for rank in [1, 2, 3]:
    case('conv' + str(rank) + 'd', [node('Conv', ['X', 'W', 'B'], pads=[1] * (rank * 2), strides=[2] * rank, group=2)],
         {'X': rand(1, 4, *([5] * rank))}, {'W': rand(6, 2, *([3] * rank)), 'B': rand(6)})
case('conv_dilated', [node('Conv', ['X', 'W'], pads=[1, 0, 0, 2], dilations=[2, 1])], {'X': rand(1, 2, 5, 6)}, {'W': rand(3, 2, 2, 3)})
case('conv_same', [node('Conv', ['X', 'W'], strides=[2, 2], auto_pad='SAME_LOWER')], {'X': rand(1, 2, 6, 5)}, {'W': rand(3, 2, 3, 3)})
case('conv_transpose', [node('ConvTranspose', ['X', 'W', 'B'], strides=[2, 2], pads=[1, 0, 0, 1], group=2, output_padding=[1, 0])],
     {'X': rand(1, 4, 3, 4)}, {'W': rand(4, 3, 3, 2), 'B': rand(6)})
for op in ['MaxPool', 'AveragePool']:
    outputs = ['Y', 'I'] if op == 'MaxPool' else ['Y']
    case(op.lower(), [node(op, ['X'], outputs, kernel_shape=[3, 2], strides=[2, 2], pads=[1, 1, 1, 0], ceil_mode=1)],
         {'X': rand(1, 2, 4, 5)}, outputs=outputs)
case('averagepool_include', [node('AveragePool', ['X'], kernel_shape=[3, 3], strides=[2, 2], pads=[1]*4, ceil_mode=1, count_include_pad=1)], {'X': rand(1, 1, 4, 4)})
for op in ['GlobalAveragePool', 'GlobalMaxPool']:
    case(op.lower(), [node(op, ['X'])], {'X': rand(2, 3, 4, 5)})
for mode in ['constant', 'reflect', 'edge', 'wrap']:
    weights = {'P': np.array([0, -1, 1, 2], np.int64)}
    case('pad_' + mode, [node('Pad', ['X', 'P'], mode=mode)], {'X': rand(3, 4)}, weights, opset=19)
for mode, coordinate, nearest in [('nearest', 'asymmetric', 'floor'), ('nearest', 'half_pixel', 'round_prefer_floor'),
                                   ('linear', 'half_pixel', 'round_prefer_floor'), ('linear', 'align_corners', 'floor')]:
    case('resize_' + mode + '_' + coordinate, [node('Resize', ['X', '', '', 'S'], mode=mode, coordinate_transformation_mode=coordinate, nearest_mode=nearest)],
         {'X': rand(1, 1, 3, 4)}, {'S': np.array([1, 1, 5, 7], np.int64)})
# RNN variants independently exercised on ONNX Runtime; reference evaluator handles layout=1.
for op, gates in [('RNN', 1), ('GRU', 3), ('LSTM', 4)]:
    for direction in ['forward', 'reverse', 'bidirectional']:
        d, n, t, i, hidden = (2 if direction == 'bidirectional' else 1), 2, 3, 2, 3
        weights = {'W': rand(d, gates*hidden, i), 'R': rand(d, gates*hidden, hidden),
                   'B': rand(d, 2*gates*hidden), 'L': np.array([3, 2], np.int32), 'H': rand(d, n, hidden)}
        args = ['X', 'W', 'R', 'B', 'L', 'H']
        outputs = ['Y', 'Hn']
        if op == 'LSTM':
            weights.update(C=rand(d, n, hidden), P=rand(d, 3*hidden))
            args += ['C', 'P']
            outputs += ['Cn']
        case(op.lower() + '_' + direction, [node(op, args, outputs, direction=direction, hidden_size=hidden)], {'X': rand(t, n, i)}, weights, outputs)
case('gru_reset', [node('GRU', ['X', 'W', 'R', 'B'], ['Y', 'H'], hidden_size=2, linear_before_reset=1)],
     {'X': rand(3, 1, 2)}, {'W': rand(1, 6, 2), 'R': rand(1, 6, 2), 'B': rand(1, 12)}, outputs=['Y', 'H'])
# Small attention block: projection, scaled attention, residual, layer norm, exact GELU.
case('transformer', [node('MatMul', ['X', 'WQ'], ['Q']), node('MatMul', ['X', 'WK'], ['K']),
      node('Transpose', ['K'], ['KT'], perm=[0, 2, 1]), node('MatMul', ['Q', 'KT'], ['S']),
      node('Mul', ['S', 'Scale'], ['SS']), node('Softmax', ['SS'], ['A'], axis=-1),
      node('MatMul', ['X', 'WV'], ['V']), node('MatMul', ['A', 'V'], ['AV']), node('Add', ['AV', 'X'], ['R']),
      node('LayerNormalization', ['R', 'Gain', 'Bias'], ['N']), node('Gelu', ['N'])],
     {'X': rand(1, 3, 4)}, {'WQ': rand(4, 4), 'WK': rand(4, 4), 'WV': rand(4, 4),
      'Scale': np.array(.5, np.float32), 'Gain': np.ones(4, np.float32), 'Bias': np.zeros(4, np.float32)}, opset=20)
# Graph captures and local functions.
then = h.make_graph([node('Add', ['X', 'One'])], 'yes', [], [h.make_tensor_value_info('Y', T.FLOAT, [2])])
other = h.make_graph([node('Sub', ['X', 'One'])], 'no', [], [h.make_tensor_value_info('Y', T.FLOAT, [2])])
for condition in [True, False]:
    case('if_' + str(condition).lower(), [node('If', ['C'], then_branch=then, else_branch=other)],
         {'X': rand(2), 'C': np.array(condition)}, {'One': np.ones(2, np.float32)})
body = h.make_graph([node('Identity', ['condition'], ['next_condition']), node('Add', ['state', 'One'], ['next_state']),
                     node('Identity', ['next_state'], ['scan'])], 'body',
                    [h.make_tensor_value_info('iteration', T.INT64, []), h.make_tensor_value_info('condition', T.BOOL, []), h.make_tensor_value_info('state', T.FLOAT, [2])],
                    [h.make_tensor_value_info('next_condition', T.BOOL, []), h.make_tensor_value_info('next_state', T.FLOAT, [2]), h.make_tensor_value_info('scan', T.FLOAT, [2])])
for count in [0, 3]:
    case('loop_' + str(count), [node('Loop', ['M', 'C', 'X'], ['Y', 'Scan'], body=body)],
         {'X': rand(2)}, {'M': np.array(count, np.int64), 'C': np.array(True), 'One': np.ones(2, np.float32)}, outputs=['Y', 'Scan'])
f = h.make_function('test.deepj', 'Double', ['X'], ['Y'], [node('Add', ['X', 'X'])], [h.make_opsetid('', 18)])
case('local_function', [h.make_node('Double', ['X'], ['Y'], domain='test.deepj')], {'X': rand(2, 3)}, functions=[f])

case('mlp_fp16', [node('MatMul', ['X', 'W'], ['A']), node('Add', ['A', 'B'])],
     {'X': rand(2, 3).astype(np.float16)}, {'W': rand(3, 4).astype(np.float16), 'B': rand(4).astype(np.float16)})
external = onnx.load(ROOT / 'mlp.onnx')
weight_file = ROOT / 'mlp.weights'
if weight_file.exists():
    weight_file.unlink()
onnx.save_model(external, ROOT / 'external.onnx', save_as_external_data=True,
                all_tensors_to_one_file=True, location='mlp.weights', size_threshold=0)
(ROOT / 'external.json').write_text((ROOT / 'mlp.json').read_text())
MANIFEST.append('external')
attribute = onnx.AttributeProto(name='alpha', ref_attr_name='slope', type=onnx.AttributeProto.FLOAT)
operation = node('LeakyRelu', ['X'], ['N'])
operation.attribute.append(attribute)
f = h.make_function('test.deepj', 'Activation', ['X'], ['Y'],
                    [operation, node('Identity', ['N'])], [h.make_opsetid('', 18)], attributes=['slope'])
case('function_attributes', [h.make_node('Activation', ['X'], ['A'], domain='test.deepj', slope=.2),
                            h.make_node('Activation', ['X'], ['B'], domain='test.deepj', slope=.4),
                            node('Add', ['A', 'B'])], {'X': rand(2, 3)}, functions=[f])

case('pool_fp16', [node('AveragePool', ['X'], kernel_shape=[2, 2], strides=[2, 2])],
     {'X': rand(1, 2, 4, 4).astype(np.float16)})
case('reduce_fp16', [node('ReduceSum', ['X', 'A'], keepdims=0)],
     {'X': rand(2, 3).astype(np.float16)}, {'A': np.array([1], np.int64)})
case('norm_fp16', [node('LayerNormalization', ['X', 'S', 'B'])],
     {'X': rand(2, 4).astype(np.float16)}, {'S': rand(4).astype(np.float16), 'B': rand(4).astype(np.float16)})

then = h.make_graph([node('Add', ['X', 'X'], ['Z'])], 'then', [], [h.make_tensor_value_info('Z', T.FLOAT, [2])])
other = h.make_graph([node('Identity', ['X'], ['Z'])], 'else', [], [h.make_tensor_value_info('Z', T.FLOAT, [2])])
f = h.make_function('test.deepj', 'Conditional', ['X', 'C'], ['Y'],
                    [node('If', ['C'], then_branch=then, else_branch=other)], [h.make_opsetid('', 18)])
case('function_capture', [h.make_node('Conditional', ['X', 'C'], ['Y'], domain='test.deepj')],
     {'X': rand(2), 'C': np.array(True)}, functions=[f])
# Mixed floating and exact integer loop state verifies heterogeneous variadic contracts.
body = h.make_graph([node('Identity', ['c'], ['c_next']), node('Add', ['x', 'One'], ['x_next']),
                     node('Add', ['count', 'Increment'], ['count_next'])], 'mixed',
                    [h.make_tensor_value_info('i', T.INT64, []), h.make_tensor_value_info('c', T.BOOL, []),
                     h.make_tensor_value_info('x', T.FLOAT, [2]), h.make_tensor_value_info('count', T.INT64, [])],
                    [h.make_tensor_value_info('c_next', T.BOOL, []), h.make_tensor_value_info('x_next', T.FLOAT, [2]),
                     h.make_tensor_value_info('count_next', T.INT64, [])])
case('loop_mixed', [node('Loop', ['M', 'C', 'X', 'Start'], ['Y', 'Count'], body=body)],
     {'X': rand(2)}, {'M': np.array(2, np.int64), 'C': np.array(True), 'One': np.ones(2, np.float32),
                      'Increment': np.array(1, np.int64), 'Start': np.array(2**40 + 1, np.int64)}, outputs=['Y', 'Count'])

operation = node('LeakyRelu', ['X'], ['N'])
operation.attribute.append(onnx.AttributeProto(name='alpha', ref_attr_name='slope', type=onnx.AttributeProto.FLOAT))
f = h.make_function('test.deepj', 'Activation', ['X'], ['Y'], [operation, node('Identity', ['N'])],
                    [h.make_opsetid('', 18)], attributes=['slope'])
case('function_hygiene', [node('Identity', ['X'], ['__deepj_function_0_N']),
      h.make_node('Activation', ['X'], ['A'], domain='test.deepj', slope=.2), node('Add', ['A', '__deepj_function_0_N'])],
     {'X': rand(2)}, functions=[f])
(ROOT / 'manifest.json').write_text(json.dumps(dict(onnx=onnx.__version__, onnxruntime=ort.__version__, fixtures=MANIFEST), indent=2) + '\n')
print('Fixtures:', len(MANIFEST))
