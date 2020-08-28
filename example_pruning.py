# type: ignore

import minituna_v3 as minituna

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise minituna.TrialPruned()

    accuracy = clf.score(valid_x, valid_y)
    return 1 - accuracy


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 30)

    best_trial = study.best_trial
    print(
        f"Best trial: id={best_trial.trial_id} value={best_trial.value} params={best_trial.params}"
    )
