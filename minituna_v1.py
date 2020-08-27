import random

from dataclasses import dataclass, field
from typing import Callable
from typing import Dict
from typing import List


@dataclass
class Distribution:
    low: float
    high: float


@dataclass
class FrozenTrial:
    trial_id: int
    state: str  # 'running', 'completed' or 'failed'
    value: float = 0
    params: Dict[str, float] = field(default_factory=dict)
    distributions: Dict[str, Distribution] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return self.state != "running"


class Storage:
    def __init__(self) -> None:
        self.trials: List[FrozenTrial] = []

    def create_new_trial_id(self) -> int:
        trial_id = len(self.trials)
        trial = FrozenTrial(trial_id=trial_id, state="running")
        self.trials.append(trial)
        return trial_id

    def get_trial(self, trial_id: int) -> FrozenTrial:
        return self.trials[trial_id]

    def get_best_trial(self) -> FrozenTrial:
        completed_trials = [t for t in self.trials if t.state == "completed"]
        return min(completed_trials, key=lambda t: t.value)

    def set_trial_value(self, trial_id: int, value: float):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.value = value

    def set_trial_state(self, trial_id: int, state: str):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.state = state

    def set_trial_param(
        self, trial_id: int, name: str, distribution: Distribution, value: float
    ):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.distributions[name] = distribution
        trial.params[name] = value


class Trial:
    def __init__(self, study: "Study", trial_id: int):
        self.study = study
        self.trial_id = trial_id
        self.state = "running"

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        distribution = Distribution(low=low, high=high)
        trial = self.study.storage.get_trial(self.trial_id)
        param = self.study.sampler.sample_independent(
            self.study, trial, name, distribution
        )
        self.study.storage.set_trial_param(self.trial_id, name, distribution, param)
        return param


class Study:
    def __init__(self):
        self.storage = Storage()
        self.sampler = RandomSampler()

    def optimize(self, objective: Callable[[Trial], float], n_trials: int) -> None:
        for _ in range(n_trials):
            trial_id = self.storage.create_new_trial_id()
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
    def best_trial(self):
        return self.storage.get_best_trial()


class RandomSampler:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        name: str,
        distribution: Distribution,
    ) -> float:
        return self.rng.uniform(distribution.low, distribution.high)


def create_study() -> Study:
    return Study()
