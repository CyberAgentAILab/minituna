import minituna


def objective(trial: minituna.Trial) -> float:
    x = trial.suggest_uniform("x", 0, 10)
    y = trial.suggest_uniform("y", 0, 10)
    value = (x - 3) ** 2 + (y - 5) ** 2

    z = trial.suggest_categorical("z", ["foo", "bar"])
    if z == "foo":
        value += 1
    else:
        value += 2
    return value


if __name__ == "__main__":
    study = minituna.create_study()
    study.optimize(objective, 10)

    best_trial = study.best_trial
    print(
        f"Best trial: id={best_trial.trial_id} value={best_trial.value} params={best_trial.params}"
    )
