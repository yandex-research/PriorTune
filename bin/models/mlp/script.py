"""Replay a physical MLP pack and publish only the requested member."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '8'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['MODULARML_SUPPRESS_SUMMARY_PRINT']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _common import *
import numpy as np

a = arguments('mlp')
try:
    import torch
    import lib.util
    from bin.tabpack_independent_mlp import train_one
    lib.util.configure_logging()
    lib.util.configure_torch()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This reference requires a CUDA GPU with bfloat16 support')
    if any(k.startswith('TABPACK_MLP_DEBUG_') for k in os.environ):
        raise RuntimeError('Debug overrides are incompatible with reference reproduction')
    c = json.loads(a.context.read_text())
    members = {m['config_id']: m for m in c['members']}
    if a.dataset.name != c['experiment']['dataset'] or members.get(a.config_id) != a.config:
        raise ValueError('A matching frozen physical context is required for this dataset/config')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); data = tmp / 'data'; data.mkdir(); run = tmp / 'run'; run.mkdir()
        for name in ('info.json','x_num.npy','x_bin.npy','x_cat.npy','y.npy'):
            if (a.dataset/name).exists(): shutil.copyfile(a.dataset/name, data/name)
        (data/'splits/default').mkdir(parents=True)
        for part in ('train','val','test'):
            np.save(data/f'splits/default/{part}.npy', np.asarray(np.load(a.dataset/f'splits/default/{part}.npy'),dtype=np.int32))
        c['data']['path'] = str(data)
        import lib.experiment
        lib.experiment._run(run)
        report = train_one(c,run)
        record = next(r for r in json.loads((run/'experiments.json').read_text()) if r['config_id']==a.config_id)
        if not record['status'].startswith('done_'):
            finish(a, record, error=record['status']); raise SystemExit(1)
        row = record.pop('prediction_row')
        predictions = {k:v[row] for k,v in np.load(run/'member_predictions.npz').items()}
        record.pop('prediction_artifact',None)
        record.update({'model':'mlp', 'gpu':torch.cuda.get_device_name(),
            'precision':'bfloat16', 'metric_direction':('maximize' if json.loads((a.dataset/'info.json').read_text())['task']['score'] in {'roc-auc','accuracy'} else 'minimize'),
            'score_direction':'maximize', 'physical_pack_size':len(c['members']),
            'physical_pack_seed':c['seed'], 'exported_config_ids':[a.config_id],
            'historical_checkpoint_restoration':'advanced-index-copy behavior preserved; predictions are from stopping state'})
        finish(a,record,predictions)
except Exception as exc:
    finish(a, {'model':'mlp'}, error=type(exc).__name__)
    raise
