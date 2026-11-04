"""Create an explicitly chosen physical pack for another stored dataset.

A new pack defines a new execution context; it does not reproduce an archived
pack with a different member list, physical ID, or seed.
"""
import argparse
import json
import runpy
from pathlib import Path
import numpy as np
BIN = Path(__file__).resolve().parents[2]
nn_batch_size_for_train_size = runpy.run_path(str(BIN / 'priors/tabicl-v2.py'))['nn_batch_size_for_train_size']
make_pack_config = runpy.run_path(str(
    Path(__file__).resolve().with_name('sample_hp.py')
))['make_pack_config']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--start',type=int,required=True)
p.add_argument('--stop',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if not 0 <= a.start < a.stop <= 4000: p.error('Require 0 <= start < stop <= 4000')
n=len(np.load(a.dataset/'splits/default/train.npy'))
c=make_pack_config(dataset=a.dataset.name,dataset_path=Path(a.dataset.name),
    config_ids=list(range(a.start,a.stop)),batch_size=nn_batch_size_for_train_size(n),
    physical_pack_id=f'cfg-{a.start:04d}-{a.stop-1:04d}')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
