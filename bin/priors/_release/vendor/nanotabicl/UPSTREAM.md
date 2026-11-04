# NanoTabICL (vendored)

- Upstream: https://github.com/soda-inria/nanotabicl
- Commit: `ead24e3a1bf2dcda3ec46041a5189c034b6c0a62`
  ("add nanoprior, fix error in model preprocessing")
- Vendored on: 2026-08-10
- License: BSD 3-Clause (see `LICENSE`), Copyright (c) 2025, Soda team @ Inria.

## Contents

| File | Status |
| --- | --- |
| `prior.py` | byte-identical to upstream |
| `model.py` | byte-identical to upstream |
| `LICENSE` | byte-identical to upstream |
| `__init__.py` | added here (empty, makes the directory importable) |
| `UPSTREAM.md` | added here |

No upstream file is modified in any way. The project-side wrapper lives in
`lib/prior.py`; in particular, the bounded replacement for the unbounded
rejection loop of `rand_dataset_filtered` is implemented there by calling
`rand_dataset_plain` per attempt and applying the same acceptance check.

## How to update

```
git clone https://github.com/soda-inria/nanotabicl.git /tmp/nanotabicl-upstream
cp /tmp/nanotabicl-upstream/{prior.py,model.py,LICENSE} vendor/nanotabicl/
```

Then update the commit hash above and re-run the tests
(a parity test in the generator's development repository, not shipped).
