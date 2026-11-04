from pathlib import Path

def get_project_dir(): return Path(__file__).resolve().parents[1]
def get_exp_dir(): return get_project_dir() / 'exp'
def get_cache_dir():
    p = get_project_dir() / 'cache'; p.mkdir(exist_ok=True); return p
def get_data_dir(): return get_project_dir() / 'data'
def get_local_dir(): return get_project_dir() / 'local'
def get_snapshot_dir(): return None
def get_tmp_output_dir(): return None
def is_local(): return True
