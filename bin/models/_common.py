"""Local-file adapters around the frozen experiment implementations."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

BIN = Path(__file__).resolve().parents[1]
RUNTIME = Path(__file__).resolve().parent / '_runtime'
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(BIN))
import artifacts  # noqa: E402

def arguments(model):
    p = argparse.ArgumentParser(description=f'Train {model} on one stored default split.')
    p.add_argument('--dataset', type=Path, required=True,
                   help='a dataset directory written by bin/priors/tabicl-v2.py')
    p.add_argument('--config-id', type=int, required=True,
                   help='an id of the bank (artifacts configs/, python bin/artifacts.py download configs)')
    p.add_argument('--bank', type=Path,
                   help='a bank of your own (JSON lines, e.g. extended by sample_hp.py --extend)')
    p.add_argument('--output', type=Path, default=Path('local/fits'))
    if model == 'mlp':
        p.add_argument('--context', type=Path, required=True,
                       help='a physical pack built by bin/models/mlp/make_context.py')
    a = p.parse_args()
    a.dataset = a.dataset.resolve()
    if a.bank:
        configs = [json.loads(line) for line in a.bank.read_text().splitlines() if line.strip()]
        if model == 'lgbm':
            configs = [dict(c, tabprep=None, tabprep_enabled=False) for c in configs]
    else:
        configs = artifacts.configs(model)
    matches = [c for c in configs if c['config_id'] == a.config_id]
    if len(matches) != 1: p.error('Config ID must be in the bank')
    a.config = matches[0]
    a.destination = a.output.resolve() / model / a.dataset.name / str(a.config_id)
    if a.destination.exists(): p.error('Output already exists; choose a fresh --output directory')
    return a

def hashes(dataset):
    return {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(dataset.rglob('*')) if p.is_file() and (p.suffix == '.npy' or p.name == 'info.json')}

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')

def finish(a, report, predictions=None, error=None):
    dest = a.destination
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / 'result'; stage.mkdir()
        if predictions is not None:
            import numpy as np
            np.savez_compressed(stage / 'predictions.npz', **predictions)
        report.update({'provisional': True, 'dataset': a.dataset.name, 'split': 'default',
                       'fold': 0, 'config_id': a.config_id, 'config': a.config,
                       'dataset_files_sha256': hashes(a.dataset)})
        save_json(stage / 'report.json', report)
        save_json(stage / 'status.json', {'status': 'FAILED' if error else 'DONE',
                                        'provisional': True, **({'error': error} if error else {})})
        (stage / ('FAILED' if error else 'DONE')).touch()
        stage.rename(dest)
    print(dest)

def cpu_main(model):
    a = arguments(model)
    try:
        import autogluon
        autogluon.__path__ = [str(RUNTIME / 'ag_overlay/autogluon'), *autogluon.__path__]
        import autogluon.common
        autogluon.common.__path__ = [str(RUNTIME / 'ag_overlay/autogluon/common'), *autogluon.common.__path__]
        import cpu
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            cpu.OUTPUT_ROOT = Path(tmp)
            result = cpu.run_config(benchmark='tabicl-v2', dataset=a.dataset.name,
                dataset_root=a.dataset, config=a.config, num_cpus=8)
            if result['state'] != 'done':
                raise RuntimeError(result['failure']['error'])
            report = result['report']; report.pop('dataset_root', None)
            predictions = dict(np.load(Path(result['path']) / 'predictions.npz'))
            report['model'] = model
            report['metric_direction'] = 'maximize' if report['score_name'] in {'roc-auc','accuracy'} else 'minimize'
            report['prediction_type'] = 'raw regression values' if report['task_type'] == 'regression' else 'probabilities'
            finish(a, report, predictions)
    except Exception as exc:
        # Do not publish host paths or tracebacks with environment information.
        finish(a, {'model': model}, error=type(exc).__name__)
        raise
