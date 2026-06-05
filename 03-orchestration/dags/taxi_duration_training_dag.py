from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import tempfile
import os

# Importa le tue funzioni
from train import read_dataframe, preprocessing, hyperparam_optimization, train_best_model, save_model

TMP_DIR = '/opt/airflow/pipeline_tmp'  # cartella condivisa tra i task

default_args = {
    'owner': 'mattfelici',
    'retries': 1,
}

with DAG(
    dag_id='nyt_taxi_duration_training',
    default_args=default_args,
    schedule=None,               # esecuzione manuale, o metti un cron
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={                     # parametri passabili al trigger
        'year': '2024',
        'month': '01',
        'n_trials': 10,
        'num_boost_round': 100,
    },
    tags=['ml', 'taxi'],
) as dag:

    def _read_dataframe(**context):
        p = context['params']
        os.makedirs(TMP_DIR, exist_ok=True)
        df = read_dataframe(p['year'], p['month'])
        df.to_parquet(f'{TMP_DIR}/df.parquet')

    def _preprocessing(**context):
        import pandas as pd
        categorical = ['locationID', 'dow', 'pickup_hour']
        numerical = ['trip_distance']

        df = pd.read_parquet(f'{TMP_DIR}/df.parquet')
        train, val, y_train, y_val, dv = preprocessing(
            df, categorical, numerical)

        train.save_binary(f'{TMP_DIR}/train.dmatrix')
        val.save_binary(f'{TMP_DIR}/val.dmatrix')

        with open(f'{TMP_DIR}/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(f'{TMP_DIR}/y_val.pkl',   'wb') as f:
            pickle.dump(y_val, f)
        with open(f'{TMP_DIR}/dv.pkl',       'wb') as f:
            pickle.dump(dv, f)

    def _hyperparam_optimization(**context):
        import xgboost as xgb
        p = context['params']

        train = xgb.DMatrix(f'{TMP_DIR}/train.dmatrix')
        val = xgb.DMatrix(f'{TMP_DIR}/val.dmatrix')
        with open(f'{TMP_DIR}/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(f'{TMP_DIR}/y_val.pkl',   'rb') as f:
            y_val = pickle.load(f)

        run_name = f"XGB_optuna_{p['year']}_{p['month']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        best_params = hyperparam_optimization(
            train, val, y_train, y_val, run_name,
            n_trials=p['n_trials'], num_boost_round=p['num_boost_round']
        )
        with open(f'{TMP_DIR}/best_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)

    def _train_best_model(**context):
        import xgboost as xgb
        p = context['params']

        train = xgb.DMatrix(f'{TMP_DIR}/train.dmatrix')
        val = xgb.DMatrix(f'{TMP_DIR}/val.dmatrix')
        with open(f'{TMP_DIR}/y_train.pkl',    'rb') as f:
            y_train = pickle.load(f)
        with open(f'{TMP_DIR}/y_val.pkl',      'rb') as f:
            y_val = pickle.load(f)
        with open(f'{TMP_DIR}/best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)

        run_name = f"XGB_train_{p['year']}_{p['month']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        booster = train_best_model(
            train, val, y_train, y_val, best_params, run_name,
            num_boost_round=p['num_boost_round']
        )
        booster.save_model(f'{TMP_DIR}/booster.json')

    def _save_model(**context):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(f'{TMP_DIR}/booster.json')
        with open(f'{TMP_DIR}/dv.pkl', 'rb') as f:
            dv = pickle.load(f)
        save_model(dv, booster)

    # --- definizione dei task ---

    t1 = PythonOperator(task_id='read_dataframe',
                        python_callable=_read_dataframe)
    t2 = PythonOperator(task_id='preprocessing',
                        python_callable=_preprocessing)
    t3 = PythonOperator(task_id='hyperparam_optimization',
                        python_callable=_hyperparam_optimization)
    t4 = PythonOperator(task_id='train_best_model',
                        python_callable=_train_best_model)
    t5 = PythonOperator(task_id='save_model',
                        python_callable=_save_model)

    t1 >> t2 >> t3 >> t4 >> t5
