"""Regenerate the 4,000-config LightGBM bank shared by lgbm and lgbm-prep
(artifacts configs/lgbm-4000.jsonl), check it against the published one, or extend it."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _sample_hp import extend, generate, save_or_check
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, help='write the bank here instead of checking it')
p.add_argument('--extend', type=int, metavar='N',
               help='write a bank of N configs: the published 4,000 and new ids after them')
a = p.parse_args()
save_or_check('lgbm-prep', extend(a.extend) if a.extend else generate(), a.output)
