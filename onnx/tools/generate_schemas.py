"""Pin official ONNX operator contracts used by the importer (no Python runtime dependency)."""
from pathlib import Path
import re
import onnx
from onnx import defs, TensorProto

ROOT = Path(__file__).resolve().parents[1]
source = '\n'.join(p.read_text() for p in (ROOT / 'src/main/java').rglob('*.java'))
names = set(re.findall(r'"([A-Z][A-Za-z0-9]+)"', source))
# Registration helpers use a name argument instead of a literal add call.
names.update(re.findall(r'(?:binary|unary)\(r,\s*"([A-Za-z0-9]+)"', source))
types = {TensorProto.DataType.Name(i).lower(): i for i in range(1, 27)}
def accepted(type_string, constraints):
    allowed = constraints.get(type_string, [type_string])
    return ','.join(str(types[t[7:-1]]) for t in allowed if t.startswith('tensor(') and t[7:-1] in types)

lines = ['# ONNX ' + onnx.__version__ + '; generated from official operator schemas. Apache-2.0.']
for s in sorted(defs.get_all_schemas_with_history(), key=lambda s: (s.name, s.since_version)):
    if s.domain or s.name not in names:
        continue
    constraints = {c.type_param_str: c.allowed_type_strs for c in s.type_constraints}
    attributes = ','.join(f'{a.name}:{a.type.value}:{int(a.required)}' for a in sorted(s.attributes.values(), key=lambda a: a.name))
    inputs = ';'.join(f'{p.type_str}:{p.option.value}:{accepted(p.type_str, constraints)}:{int(p.is_homogeneous)}' for p in s.inputs) or "-"
    lines.append('\t'.join(map(str, [s.name, s.since_version, s.min_input, s.max_input, s.min_output, s.max_output, attributes, inputs])))
path = ROOT / 'src/main/resources/io/github/kirstenali/deepj/onnx/schemas.tsv'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('\n'.join(lines) + '\n')
print(len(lines) - 1, 'versioned contracts;', len(names), 'operators')
