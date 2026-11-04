"""Parity with the implementation that produced the published results.

The fixtures under fixtures/ hold small inputs and the outputs the original
(cluster-scale) implementation computed from them; each test recomputes the outputs
with this directory's code: selections, credits, worlds and deployed scores bit for
bit, means over repetitions to rounding. No GPU, no artifacts, a few seconds:

    python -m unittest discover -s bin/hillclimb/tests
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core                                                         # noqa: E402
import ensemble as te                                               # noqa: E402
import grid                                                         # noqa: E402
import portfolio as selection                                       # noqa: E402
import worlds                                                       # noqa: E402

torch.set_grad_enabled(False)
FIXTURES = HERE / 'fixtures'


def arrays(z, prefix):
    return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def quiet(*args, **kwargs):
    pass


class TStudy(unittest.TestCase):
    """48 synthetic datasets x 64 configs, R = 4 worlds, K = 5, three repetitions."""

    @classmethod
    def setUpClass(cls):
        z = np.load(FIXTURES / 't_study.npz')
        cls.z = z
        cls.synth, cls.real = core.load(arrays(z, 'synth_')), core.load(arrays(z, 'real_'))
        cls.record = json.loads((FIXTURES / 't_study.json').read_text())

    def check(self, tag, full_pool):
        z = self.z
        out = grid.surfaces(self.synth, self.real, z['delta'], (4242,), 5, 3, 4,
                            full_pool=full_pool, log=quiet)
        cfg, data = core.shuffles(0, (4242,), 64, 48, 3)
        np.testing.assert_array_equal(cfg, z[f'{tag}_config_orders'])
        np.testing.assert_array_equal(data, z[f'{tag}_dataset_orders'])
        np.testing.assert_array_equal(core.references(0, (4242,), cfg, 5, 4), z[f'{tag}_selection_bank'])
        np.testing.assert_array_equal(out['grid_c'], z[f'{tag}_grid'])
        for arm in ('base', 'val'):
            np.testing.assert_array_equal(out[f'orders_{arm}'], z[f'{tag}_orders_{arm}'])
            np.testing.assert_array_equal(out[f'objective_{arm}'], z[f'{tag}_objective_{arm}'])
            np.testing.assert_array_equal(out[f'scores_{arm}'], z[f'{tag}_T_scores_{arm}'])
            # Means over repetitions agree to rounding: numpy's summation order depends
            # on memory alignment, so not even the original reproduces its last bit.
            mean, ci = core.summarize(out[f'metrics_{arm}'])
            np.testing.assert_allclose(mean, self.record[tag][arm]['metrics'], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(ci, self.record[tag][arm]['ci95'], rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(out['reference_scores'], z[f'{tag}_T_reference_scores'])

    def test_grid(self):
        self.check('grid', False)

    def test_full_pool(self):
        self.check('full', True)

    def test_portfolio_is_the_full_pool_cell_of_repetition_zero(self):
        order, _ = selection.select_t(self.synth, self.z['delta'], (4242,), 5, 4)
        np.testing.assert_array_equal(order, self.z['full_orders_val'][0, -1, -1])

    def test_selection_rules(self):
        """The diagnostic's four rules: plain, validation worlds only, references only,
        and paired; each a choice of world and reference axes for the same climb."""
        z = self.z
        cfg, _ = core.shuffles(0, (4242,), 64, 48, 2)
        banks = core.references(0, (4242,), cfg, 5, 4)
        np.testing.assert_array_equal(banks, z['diag_banks'])
        split = z['diag_folds']
        base = core.world_ranks(self.synth.val, None, self.synth.available)
        boot = core.world_ranks(self.synth.val, z['delta'], self.synth.available)
        rules = [(1, False, False)] + [(n, v, r) for n in (2, 4)
                                       for v, r in ((True, False), (False, True), (True, True))]
        for fold in range(2):
            train = np.flatnonzero(split != fold)
            for seed in range(2):
                order = core.rng(0, 991, seed).permutation(4)
                for m, (count, val_boot, ref_average) in enumerate(rules):
                    ranks = boot[..., order[:count]] if val_boot else base
                    refs = banks[seed, :count if ref_average else 1]
                    r, credit, empty = core.credit_tensors(self.synth, ranks, refs)
                    chosen, _ = core.climb(r, credit, empty, np.arange(64), train, 5)
                    np.testing.assert_array_equal(chosen, z['diag_orders'][m, fold, seed])


class TRealTies(unittest.TestCase):
    """200 datasets x 400 configs of a real LightGBM-TabPrep score cache: real ties."""

    def test_grid(self):
        z = np.load(FIXTURES / 't_real.npz')
        synth, real = core.load(arrays(z, 'synth_')), core.load(arrays(z, 'real_'))
        i, c, r = np.meshgrid(np.arange(200), np.arange(400), np.arange(16), indexing='ij')
        delta = ((i * 7919 + c * 104729 + r * 15485863) % 2001 - 1000) * 1e-5 * np.abs(
            np.nan_to_num(synth.val))[..., None]
        delta = np.where(synth.available[..., None], delta, np.nan).astype(np.float32)
        out = grid.surfaces(synth, real, delta, (827,), 25, 2, 16, log=quiet)
        for arm in ('base', 'val'):
            np.testing.assert_array_equal(out[f'orders_{arm}'], z[f'orders_{arm}'])
            np.testing.assert_array_equal(out[f'objective_{arm}'], z[f'objective_{arm}'])
        np.testing.assert_array_equal(out['scores_val'], z['T_scores_val'])


class Predictions(unittest.TestCase):
    """Six prediction sets (two per task type), seven configs, two failed cells."""

    @classmethod
    def setUpClass(cls):
        z = np.load(FIXTURES / 'predictions.npz')
        cls.z = z
        cls.cache = core.load(arrays(z, 'pred_scores_'))
        cls.preds = [{name: z[f'pred{i}_{name}'] for name in
                      ('val', 'test', 'y_val', 'y_test', 'available')} for i in range(6)]

    def dataset(self, i):
        p = self.preds[i]
        return te.prepare(p['val'], p['test'], p['y_val'], p['y_test'], str(self.cache.metrics[i]),
                          p['available'], self.cache.test[i])

    def test_worlds(self):
        for i, p in enumerate(self.preds):
            delta, _ = worlds.dataset_delta(p['val'], p['y_val'], str(self.cache.metrics[i]),
                                            p['available'], 4, 0, (827,), i)
            np.testing.assert_array_equal(delta, self.z['worlds_delta'][i])

    def test_te_deployment(self):
        ids = self.z['te_ids']
        for i in range(6):
            ds = self.dataset(i)
            loss = np.asarray([te.deploy(ds, p) for p in ids])
            raw = -loss if self.cache.higher[i] else loss
            np.testing.assert_allclose(raw, self.z['te_scores'][:, i], rtol=1e-12)

    def test_te_objective(self):
        datasets = [self.dataset(i) for i in range(6)]
        resamples = [selection.bootstrap_rows(p['y_val'], str(self.cache.metrics[i]), 4, 0, (827,), i)
                     for i, p in enumerate(self.preds)]
        cfg, _ = core.shuffles(0, (827,), 7, 6, 1)
        bank = core.references(0, (827,), cfg, 3, 4)[0]
        np.testing.assert_array_equal(bank, self.z['teo_bank'])
        order, history = selection.select_te(datasets, resamples, bank, 4, log=quiet)
        np.testing.assert_array_equal(order, self.z['teo_order'])
        for step, credits in enumerate(history):
            np.testing.assert_array_equal(credits, self.z['teo_credit'][step][:len(credits)])


if __name__ == '__main__':
    unittest.main()
