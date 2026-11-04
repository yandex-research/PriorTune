import os, sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _common import cpu_main
cpu_main('lgbm-prep')
