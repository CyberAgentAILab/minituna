import abc
import copy
import math
import random

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

CategoricalChoiceType = Union[None, bool, int, float, str]


class BaseDistribution(abc.ABC):
    @abc.abstractmethod
    def to_internal_repr(self, external_repr: Any) -> float:
        ...

    @abc.abstractmethod
    def to_external_repr(self, internal_repr: float) -> Any:
        ...


class UniformDistribution(BaseDistribution):
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def to_internal_repr(self, external_repr: Any) -> float:
        return external_repr

    def to_external_repr(self, internal_repr: float) -> Any:
        return internal_repr


class LogUniformDistribution(BaseDistribution):
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def to_internal_repr(self, external_repr: Any) -> float:
        return external_repr

    def to_external_repr(self, internal_repr: float) -> Any:
        return internal_repr


class IntUniformDistribution(BaseDistribution):
    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def to_internal_repr(self, external_repr: Any) -> float:
        return float(external_repr)

    def to_external_repr(self, internal_repr: float) -> Any:
        return int(internal_repr)


class CategoricalDistribution(BaseDistribution):
    def __init__(self, choices: List[CategoricalChoiceType]) -> None:
        self.choices = choices

    def to_internal_repr(self, external_repr: Any) -> float:
        return self.choices.index(external_repr)

    def to_external_repr(self, internal_repr: float) -> Any:
        return self.choices[int(internal_repr)]


class FrozenTrial:
    def __init__(self, trial_id: int, state: str) -> None:
        self.trial_id = trial_id
        self.state = state  # 'running', 'completed' or 'failed'
        self.value: Optional[float] = None
        self.internal_params: Dict[str, float] = {}
        self.distributions: Dict[str, BaseDistribution] = {}

    @property
    def is_finished(self) -> bool:
        return self.state != "running"

    @property
    def params(self) -> Dict[str, Any]:
        external_repr = {}
        for param_name in self.internal_params:
            distribution = self.distributions[param_name]
            internal_repr = self.internal_params[param_name]
            external_repr[param_name] = distribution.to_external_repr(internal_repr)
        return external_repr


class Storage:
    def __init__(self) -> None:
        self.trials: List[FrozenTrial] = []

    def create_new_trial(self) -> int:
        trial_id = len(self.trials)
        trial = FrozenTrial(trial_id=trial_id, state="running")
        self.trials.append(trial)
        return trial_id

    def get_trial(self, trial_id: int) -> FrozenTrial:
        return copy.deepcopy(self.trials[trial_id])

    def get_best_trial(self) -> FrozenTrial:
        completed_trials = [t for t in self.trials if t.state == "completed"]
        best_trial = min(completed_trials, key=lambda t: t.value)
        return copy.deepcopy(best_trial)

    def set_trial_value(self, trial_id: int, value: float) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.value = value

    def set_trial_state(self, trial_id: int, state: str) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.state = state

    def set_trial_param(
        self, trial_id: int, name: str, distribution: BaseDistribution, value: float
    ) -> None:
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.distributions[name] = distribution
        trial.internal_params[name] = value


class Trial:
    def __init__(self, study: "Study", trial_id: int):
        self.study = study
        self.trial_id = trial_id
        self.state = "running"

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        storage = self.study.storage

        trial = storage.get_trial(self.trial_id)
        param_value = self.study.sampler.sample_independent(
            self.study, trial, name, distribution
        )
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)
        storage.set_trial_param(
            self.trial_id, name, distribution, param_value_in_internal_repr
        )
        return param_value

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        return self._suggest(name, UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        return self._suggest(name, LogUniformDistribution(low=low, high=high))

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self._suggest(name, IntUniformDistribution(low=low, high=high))

    def suggest_categorical(
        self, name: str, choices: List[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._suggest(name, CategoricalDistribution(choices=choices))


class Sampler:
    def __init__(self, seed: int = None) -> None:
        self.rng = random.Random(seed)

    def sample_independent(
        self,
        study: "Study",
        trial: FrozenTrial,
        name: str,
        distribution: BaseDistribution,
    ) -> Any:
        if isinstance(distribution, UniformDistribution):
            return self.rng.uniform(distribution.low, distribution.high)
        elif isinstance(distribution, LogUniformDistribution):
            log_low = math.log(distribution.low)
            log_high = math.log(distribution.high)
            return math.exp(self.rng.uniform(log_low, log_high))
        elif isinstance(distribution, IntUniformDistribution):
            return self.rng.randint(distribution.low, distribution.high)
        elif isinstance(distribution, CategoricalDistribution):
            index = self.rng.randint(0, len(distribution.choices) - 1)
            return distribution.choices[index]
        else:
            raise ValueError("unsupported distribution")


class Study:
    def __init__(self, storage: Storage, sampler: Sampler) -> None:
        self.storage = storage
        self.sampler = sampler

    def optimize(self, objective: Callable[[Trial], float], n_trials: int) -> None:
        for _ in range(n_trials):
            trial_id = self.storage.create_new_trial()
            trial = Trial(self, trial_id)

            try:
                value = objective(trial)
                self.storage.set_trial_value(trial_id, value)
                self.storage.set_trial_state(trial_id, "completed")
                print(f"trial_id={trial_id} is completed with value={value}")
            except Exception as e:
                self.storage.set_trial_state(trial_id, "failed")
                print(f"trial_id={trial_id} is failed by {e}")

    @property
    def best_trial(self) -> FrozenTrial:
        return self.storage.get_best_trial()


def create_study(
    storage: Optional[Storage] = None,
    sampler: Optional[Sampler] = None,
) -> Study:
    return Study(
        storage=storage or Storage(),
        sampler=sampler or Sampler(),
    )
