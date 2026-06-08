from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pickle
import os
import logging

# Importa le tue funzioni
from train import read_dataframe, preprocessing, hyperparam_optimization, train_best_model, save_model

TMP_DIR = '/opt/airflow/pipeline_tmp'  # cartella condivisa tra i task

default_args = {
    'owner': 'mattfelici',
    'retries': 1,
}
logger = logging.getLogger('airflow.task')


with DAG(
    dag_id='nyt_taxi_duration_training',
    default_args=default_args,
    schedule=None,               # esecuzione manuale, o metti un cron
    start_date=datetime(2026, 1, 1),
    catchup=False,
    params={                     # parametri passabili al trigger
        'year': '2026',
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
        import mlflow
        
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001'))
        mlflow.set_experiment('nyt-taxi-duration')
        
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
        import mlflow
        
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001'))
        mlflow.set_experiment('nyt-taxi-duration')
        
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
        context['ti'].xcom_push(key='run_id', value=mlflow.last_active_run().info.run_id)


    def _save_model(**context):
        import mlflow
        import xgboost as xgb
        
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001'))
        mlflow.set_experiment('nyt-taxi-duration')
        
        run_id = context['ti'].xcom_pull(key='run_id', task_ids='train_best_model')
        
        with mlflow.start_run(run_id=run_id):  # riprende il run esistente
            logger.info(f'MLFLOW_TRACKING_URI env: {os.getenv("MLFLOW_TRACKING_URI")}')
            logger.info(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
            logger.info(f'MLflow artifact URI: {mlflow.get_artifact_uri()}')
            logger.info(f'Resuming run_id: {run_id}')

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
