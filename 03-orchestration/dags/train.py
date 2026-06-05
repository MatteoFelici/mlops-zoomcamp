from datetime import datetime
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pickle
import xgboost as xgb
import optuna
import mlflow


def read_dataframe(year: str,
                   month: str) -> pd.DataFrame:

    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    print(f'Reading dataframe from {data_url}')
    df = pd.read_parquet(
        data_url,
        engine='fastparquet'
    )
    print(f'{df.shape[0]} records loaded')

    # Duration
    df['duration'] = (df['tpep_dropoff_datetime'] -
                      df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df.loc[df['duration'].between(1, 60)]

    # Date info
    df['dow'] = df['tpep_pickup_datetime'].dt.day_name()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    # Pairing locations
    df['locationID'] = df['PULocationID'].astype(
        str) + '_' + df['DOLocationID'].astype(str)

    return df


def preprocessing(
    df:  pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    tgt: str = 'duration'
) -> tuple[xgb.DMatrix, xgb.DMatrix, pd.Series, pd.Series, DictVectorizer]:

    print('Start preprocessing')
    dv = DictVectorizer(separator='_')
    tmp = df[categorical + numerical]
    tmp[categorical] = tmp[categorical].astype(str)
    y = df[tgt]

    X_train, X_val, y_train, y_val = train_test_split(tmp, y, train_size=0.8)
    X_train = dv.fit_transform(X_train.to_dict(orient='records'))

    X_val = dv.transform(X_val.to_dict(orient='records'))

    # Create XGB matrices
    train = xgb.DMatrix(X_train, label=y_train.reset_index(drop=True))
    val = xgb.DMatrix(X_val, label=y_val.reset_index(drop=True))

    return train, val, y_train, y_val, dv


def hyperparam_optimization(
    train: xgb.DMatrix,
    val :xgb.DMatrix,
    y_train: pd.Series,
    y_val: pd.Series,
    run_name: str,
    n_trials,
    num_boost_round
) -> dict:
    
    print('Start optimization')

    def optimize(trial):

        with mlflow.start_run(nested=True, run_name=f'trial_{trial.number}') as child_run:
            mlflow.set_tag('dev', 'mattfelici')
            mlflow.log_param('model_type', 'XGBoost')
            mlflow.set_tag('model-optimization', 'hyperopt')

            params = {
                'max_depth': trial.suggest_int('max_depth', 4, 25),
                'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1e-1, log=True),
                'min_child_weight': trial.suggest_float('min_child_weight', 1e-1, 1e3, log=True),
                'objective': 'reg:squarederror',
                'seed': 42
            }

            # mlflow.xgboost.autolog()
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=num_boost_round,
                evals=[(val, 'validation')],
                early_stopping_rounds=50
            )
            pred_train = booster.predict(train)
            rmse_train = root_mean_squared_error(y_train, pred_train)
            mlflow.log_metric('RMSE_train', rmse_train)

            pred_val = booster.predict(val)
            rmse_val = root_mean_squared_error(y_val, pred_val)
            mlflow.log_metric('RMSE_val', rmse_val)

            trial.set_user_attr("run_id", child_run.info.run_id)

            return rmse_val

    with mlflow.start_run(run_name=run_name) as run:

        n_trials = n_trials
        mlflow.log_param('N trials', n_trials)

        study = optuna.create_study(direction="minimize")
        study.optimize(optimize, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)

    print('Best parameters found:')
    print(study.best_trial.params)
    return study.best_trial.params


def train_best_model(
    train: xgb.DMatrix,
    val: xgb.DMatrix,
    y_train: pd.Series,
    y_val: pd.Series,
    best_params: dict,
    run_name: str,
    num_boost_round: int
) -> xgb.Booster :

    print('Applying best parameters to train the main model')
    
    with mlflow.start_run(run_name=run_name) as run:

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=num_boost_round,
            evals=[(val, 'validation')],
            early_stopping_rounds=50
        )
        pred_train = booster.predict(train)
        rmse_train = root_mean_squared_error(y_train, pred_train)
        mlflow.log_metric('RMSE_train', rmse_train)

        pred_val = booster.predict(val)
        rmse_val = root_mean_squared_error(y_val, pred_val)
        mlflow.log_metric('RMSE_val', rmse_val)

        return booster


def save_model(dv: DictVectorizer,
               booster: xgb.Booster) -> None:

    print('Save artifacts')
    
    with open('dict_vectorizer_fit.b', 'wb') as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(
        'dict_vectorizer_fit.b',
        artifact_path='preprocessors'
    )

    mlflow.xgboost.log_model(booster, name='xgb_model')

    return None


def run(year: str,
        month: str,
        n_trials: int,
        num_boost_round: int) -> None:
    
    mlflow.set_experiment('nyt-taxi-duration')
    
    categorical = ['locationID', 'dow', 'pickup_hour']
    numerical = ['trip_distance']
    
    df = read_dataframe(year, month)
    
    train, val, y_train, y_val, dv = preprocessing(df, categorical, numerical)
    
    right_now = datetime.now().strftime("%Y%m%d%H%M%S")
    best_params = hyperparam_optimization(
        train, val, y_train, y_val, f'XGB_optuna_{year}_{month}_{right_now}',
        n_trials=n_trials, num_boost_round=num_boost_round
    )
    
    right_now = datetime.now().strftime("%Y%m%d%H%M%S")
    booster = train_best_model(
        train, val, y_train, y_val, best_params, f'XGB_train_{year}_{month}_{right_now}',
        num_boost_round=num_boost_round
    )
    
    save_model(dv, booster)
    
    return None
    
    
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model to predict NYC taxi trip duration')
    parser.add_argument('--year', type=int, required=True, help='Year for training data')
    parser.add_argument('--month', type=int, required=True, help='Month for training data')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials in hyperparam optimization - default 10')
    parser.add_argument('--num_boost_round', type=int, default=100, help='Number of boosting rounds for each XGB model - default 100')
    args = parser.parse_args()
    
    year_str = f'{args.year}'
    month_str = f'{args.month:02d}'
    
    run(year=year_str, month=month_str, n_trials=args.n_trials, num_boost_round=args.num_boost_round)