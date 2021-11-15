# minituna

minituna is a toy hyperparameter optimization framework intended for understanding Optuna's internal design. Required Python version is 3.8 or later (due to the use of `typing.Literal`).

## minituna_v1 (≒ 100 lines)

```python
import minituna_v1 as minituna

def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_uniform("x", 0, 10)
    y = trial.suggest_uniform("y", 0, 10)
    return (x - 3) ** 2 + (y - 5) ** 2

if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)
    best_trial = study.best_trial
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}"
    )
```

<details>
<summary>Output of `example_quadratic.py`</summary>

```console
$ python example_quadratic.py
trial_id=0 is completed with value=36.658565123549835
trial_id=1 is completed with value=36.58945605027185
trial_id=2 is completed with value=36.261419630096924
trial_id=3 is completed with value=15.904426822321941
trial_id=4 is completed with value=31.00261936939642
trial_id=5 is completed with value=0.3046670574062946
trial_id=6 is completed with value=22.093997465381495
trial_id=7 is completed with value=45.68550817426529
trial_id=8 is completed with value=21.059586293347397
trial_id=9 is completed with value=26.691576771270128
Best trial: value=0.3046670574062946 params={'x': 3.545340140826294, 'y': 4.9147287374911555}
```

</details>

## minituna_v2 : More distributions support (≒ 200 lines)

https://github.com/optuna/optuna/blob/master/examples/sklearn_simple.py

```python
import minituna_v2 as minituna

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_loguniform("svc_c", 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(
        classifier_obj, x, y, n_jobs=-1, cv=3
    )
    accuracy = score.mean()
    return 1 - accuracy


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)

    best_trial = study.best_trial
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}"
    )
```

<details>
<summary>Output of `example_sklearn.py`</summary>

```console
$ python example_sklearn.py
trial_id=0 is completed with value=0.040000000000000036
trial_id=1 is completed with value=0.6799999999999999
trial_id=2 is completed with value=0.033333333333333326
trial_id=3 is completed with value=0.040000000000000036
trial_id=4 is completed with value=0.046666666666666745
trial_id=5 is completed with value=0.6799999999999999
trial_id=6 is completed with value=0.053333333333333344
trial_id=7 is completed with value=0.6799999999999999
trial_id=8 is completed with value=0.040000000000000036
trial_id=9 is completed with value=0.6799999999999999
Best trial: value=0.033333333333333326 params={'classifier': 'RandomForest', 'rf_max_depth': 4}
```

</details>

## minituna_v3 : Pruning algorithm support (≒ 300 lines)

https://github.com/optuna/optuna/blob/master/examples/visualization/plot_study.ipynb

```python
import minituna_v3 as minituna

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


mnist = fetch_openml(name="Fashion-MNIST", version=1)
classes = list(set(mnist.target))

# For demonstrational purpose, only use a subset of the dataset.
n_samples = 4000
data = mnist.data[:n_samples]
target = mnist.target[:n_samples]

x_train, x_valid, y_train, y_valid = train_test_split(data, target)


def objective(trial):
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(
            [trial.suggest_int("n_units_l{}".format(i), 32, 64) for i in range(3)]
        ),
        learning_rate_init=trial.suggest_loguniform("lr_init", 1e-5, 1e-1),
    )

    for step in range(100):
        clf.partial_fit(x_train, y_train, classes=classes)
        accuracy = clf.score(x_valid, y_valid)
        error = 1 - accuracy

        # Report intermediate objective value.
        trial.report(error, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise minituna.TrialPruned()
    return error


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 30)

    best_trial = study.best_trial
    print(
        f"Best trial: value={best_trial.value} params={best_trial.params}"
    )
```

<details>
<summary>Output of `example_pruning.py`</summary>

```console
$ python example_pruning.py
trial_id=0 is completed with value=0.645
trial_id=1 is completed with value=0.30200000000000005
trial_id=2 is completed with value=0.885
trial_id=3 is completed with value=0.891
trial_id=4 is completed with value=0.241
trial_id=5 is completed with value=0.36
trial_id=6 is completed with value=0.30600000000000005
trial_id=7 is pruned at step=0 value=0.868
trial_id=8 is completed with value=0.20199999999999996
trial_id=9 is pruned at step=0 value=0.874
trial_id=10 is completed with value=0.31699999999999995
trial_id=11 is pruned at step=0 value=0.9
trial_id=12 is pruned at step=1 value=0.835
trial_id=13 is completed with value=0.238
trial_id=14 is completed with value=0.19799999999999995
trial_id=15 is pruned at step=0 value=0.9299999999999999
trial_id=16 is completed with value=0.22799999999999998
trial_id=17 is pruned at step=0 value=0.882
trial_id=18 is completed with value=0.256
trial_id=19 is pruned at step=0 value=0.87
trial_id=20 is pruned at step=0 value=0.864
trial_id=21 is pruned at step=11 value=0.377
trial_id=22 is completed with value=0.22799999999999998
trial_id=23 is completed with value=0.236
trial_id=24 is completed with value=0.20299999999999996
trial_id=25 is pruned at step=0 value=0.895
trial_id=26 is pruned at step=0 value=0.899
trial_id=27 is pruned at step=0 value=0.858
trial_id=28 is completed with value=0.21899999999999997
trial_id=29 is pruned at step=45 value=0.267
Best trial: value=0.19799999999999995 params={'n_units_l0': 52, 'n_units_l1': 51, 'n_units_l2': 61, 'lr_init': 0.005854153852825279}
```

</details>
