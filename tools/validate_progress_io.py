"""Regression check for transient shared-filesystem progress read failures."""
import ast
import errno
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

source = Path(__file__).with_name('eval_cvci3_protocol.py')
tree = ast.parse(source.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'read_progress_snapshots')
namespace = {'json': json, 'errno': errno}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             str(source), 'exec'), namespace)
read = namespace['read_progress_snapshots']
with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary)
    assert read(output, 4) == []
    for code in (errno.ENOENT, errno.ESTALE, errno.EAGAIN):
        with patch.object(Path, 'read_text', side_effect=OSError(code, 'simulated transient FS error')):
            assert read(output, 4) == []
    with patch.object(Path, 'read_text', side_effect=PermissionError(errno.EACCES, 'denied')):
        try:
            read(output, 1)
        except PermissionError:
            pass
        else:
            raise AssertionError('permission errors must propagate')
    (output / 'progress-rank0.json').write_text('{')
    assert read(output, 1) == []
    row = {'rank': 0, 'done': True}
    (output / 'progress-rank0.json').write_text(json.dumps(row))
    assert read(output, 1) == [row]
print('PASS progress I/O: ENOENT/ESTALE/EAGAIN retries, malformed snapshot retries, recovery, permission propagation')
