import copy
import random

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional


class Distribution:
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high


class FrozenTrial:
    def __init__(self, trial_id: int, state: str) -> None:
        self.trial_id = trial_id
        self.state = state  # 'running', 'completed' or 'failed'
        self.value: Optional[float] = None
        self.params: Dict[str, float] = {}

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
        return copy.deepcopy(self.trials[trial_id])

    def get_best_trial(self) -> FrozenTrial:
        completed_trials = [t for t in self.trials if t.state == "completed"]
        best_trial = min(completed_trials, key=lambda t: t.value)
        return copy.deepcopy(best_trial)

    def set_trial_value(self, trial_id: int, value: float):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.value = value

    def set_trial_state(self, trial_id: int, state: str):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
        trial.state = state

    def set_trial_param(self, trial_id: int, name: str, value: float):
        trial = self.trials[trial_id]
        assert not trial.is_finished, "cannot update finished trials"
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
        self.study.storage.set_trial_param(self.trial_id, name, param)
        return param


class Sampler:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def sample_independent(
        self,
        study: "Study",
        trial: FrozenTrial,
        name: str,
        distribution: Distribution,
    ) -> float:
        return self.rng.uniform(distribution.low, distribution.high)


class Study:
    def __init__(self, storage: Storage, sampler: Sampler):
        self.storage = storage
        self.sampler = sampler

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


def create_study(
    storage: Optional[Storage] = None,
    sampler: Optional[Sampler] = None,
) -> Study:
    return Study(
        storage=storage or Storage(),
        sampler=sampler or Sampler(),
    )
