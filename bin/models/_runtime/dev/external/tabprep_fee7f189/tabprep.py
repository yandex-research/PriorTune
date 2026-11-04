"""Scikit-learn TabPrep wrapper pinned to upstream commit fee7f189.

The upstream project is https://github.com/atschalz/tabprep.  This local copy
keeps the experiment independent of network access during training.  The small
wrapper mirrors the upstream implementation while the actual feature
generators remain the pinned AutoGluon layer implementations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from autogluon.features.generators import (
    ArithmeticFeatureGenerator,
    BulkFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
)

try:
    from autogluon.features.generators import GroupByFeatureGenerator
except ImportError:
    # The TabPrep paper targets an AutoGluon source checkout. Released
    # AutoGluon 1.5.0 does not re-export this not-yet-released generator.
    from autogluon.features.generators.groupby import GroupByFeatureGenerator

try:
    from autogluon.features.generators import (
        RandomSubsetFeatureCompressionGenerator,
    )
except ImportError:
    from autogluon.features.generators.rsfc import (
        RandomSubsetFeatureCompressionGenerator,
    )

from autogluon.features.generators.abstract import AbstractFeatureGenerator


_GENERATORS = {
    cls.__name__: cls
    for cls in (
        ArithmeticFeatureGenerator,
        BulkFeatureGenerator,
        CategoricalInteractionFeatureGenerator,
        GroupByFeatureGenerator,
        OOFTargetEncodingFeatureGenerator,
        RandomSubsetFeatureCompressionGenerator,
    )
}


class TabPrepFeatureGenerator(BaseEstimator, TransformerMixin):
    """Leakage-safe wrapper over the five TabPrep feature generators."""

    def __init__(
        self,
        target_type: str = 'binary',
        random_state: int | None = 0,
        use_groupby: bool = True,
        groupby_kwargs: dict[str, Any] | None = None,
        use_rsfc: bool = True,
        rsfc_kwargs: dict[str, Any] | None = None,
        use_arithmetic: bool = True,
        arithmetic_kwargs: dict[str, Any] | None = None,
        use_cat_interact: bool = True,
        cat_interact_kwargs: dict[str, Any] | None = None,
        use_oofte: bool = True,
        oofte_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.target_type = target_type
        self.random_state = random_state
        self.use_groupby = use_groupby
        self.groupby_kwargs = dict(groupby_kwargs or {})
        self.use_rsfc = use_rsfc
        self.rsfc_kwargs = dict(rsfc_kwargs or {})
        self.use_arithmetic = use_arithmetic
        self.arithmetic_kwargs = dict(arithmetic_kwargs or {})
        self.use_cat_interact = use_cat_interact
        self.cat_interact_kwargs = dict(cat_interact_kwargs or {})
        self.use_oofte = use_oofte
        self.oofte_kwargs = dict(oofte_kwargs or {})
        self.preprocessor_: AbstractFeatureGenerator | None = None
        self.is_fitted_ = False

    def _config(self) -> dict[str, Any]:
        stage: list[Any] = []
        passthrough_types = None
        if self.use_groupby:
            stage.append(['GroupByFeatureGenerator', self.groupby_kwargs])
        if self.use_rsfc:
            stage.append(
                ['RandomSubsetFeatureCompressionGenerator', self.rsfc_kwargs]
            )
        if self.use_arithmetic:
            kwargs = dict(self.arithmetic_kwargs)
            kwargs.setdefault('passthrough', True)
            stage.append(['ArithmeticFeatureGenerator', kwargs])
        categorical: list[Any] = []
        if self.use_cat_interact:
            kwargs = dict(self.cat_interact_kwargs)
            kwargs.setdefault('passthrough', True)
            categorical.append(['CategoricalInteractionFeatureGenerator', kwargs])
        if self.use_oofte:
            categorical.append(
                ['OOFTargetEncodingFeatureGenerator', self.oofte_kwargs]
            )
            passthrough_types = {'invalid_raw_types': ['category', 'object']}
        if categorical:
            stage.append(categorical)
        result: dict[str, Any] = {'ag.prep_params': [stage] if stage else []}
        if passthrough_types is not None:
            result['ag.prep_params.passthrough_types'] = passthrough_types
        return result

    def _init_generator(
        self, generator: type[AbstractFeatureGenerator] | str, kwargs: dict | None
    ) -> AbstractFeatureGenerator:
        if isinstance(generator, str):
            try:
                generator = _GENERATORS[generator]
            except KeyError as error:
                raise ValueError(f'Unknown TabPrep generator: {generator}') from error
        params = dict(kwargs or {})
        params.setdefault('verbosity', 0)
        params.setdefault('target_type', self.target_type)
        if 'random_state' not in params and self.random_state is not None:
            params['random_state'] = self.random_state
        return generator(**params)

    def _recursive_init(self, value):
        if isinstance(value, list):
            if len(value) == 2 and isinstance(
                value[0], str | AbstractFeatureGenerator
            ):
                return self._init_generator(value[0], value[1])
            return [self._recursive_init(item) for item in value]
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f'Invalid TabPrep generator tuple: {value!r}')
            return self._init_generator(value[0], value[1])
        raise ValueError(f'Invalid TabPrep generator specification: {value!r}')

    def get_preprocessor(self) -> AbstractFeatureGenerator | None:
        config = self._config()
        prep_params = config['ag.prep_params']
        if not prep_params:
            return None
        preprocessors = self._recursive_init(prep_params)
        if len(preprocessors) == 1 and isinstance(
            preprocessors[0], AbstractFeatureGenerator
        ):
            return preprocessors[0]
        return BulkFeatureGenerator(
            generators=preprocessors,
            remove_unused_features='false_recursive',
            post_drop_duplicates=True,
            passthrough=True,
            passthrough_types=config.get('ag.prep_params.passthrough_types'),
            verbosity=0,
        )

    def _reorder(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = [x for x in self.numeric_features_in_ if x in frame.columns]
        remaining = [x for x in frame.columns if x not in numeric]
        return frame.loc[:, [*numeric, *remaining]]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        del fit_params
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'X must be a DataFrame, got {type(X)}')
        if y is None:
            raise AssertionError('y is required for TabPrep')
        frame = X.copy()
        target = pd.Series(y).copy()
        self.feature_names_in_ = list(frame.columns)
        self.n_features_in_ = frame.shape[1]
        self.numeric_features_in_ = list(
            frame.select_dtypes(include='number').columns
        )
        self.preprocessor_ = self.get_preprocessor()
        output = (
            frame.copy()
            if self.preprocessor_ is None
            else self.preprocessor_.fit_transform(frame, target)
        )
        self.is_fitted_ = True
        return self._reorder(output)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise AttributeError('TabPrepFeatureGenerator is not fitted')
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'X must be a DataFrame, got {type(X)}')
        output = (
            X.copy()
            if self.preprocessor_ is None
            else self.preprocessor_.transform(X)
        )
        return self._reorder(output)
