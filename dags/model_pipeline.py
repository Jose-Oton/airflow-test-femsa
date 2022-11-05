from airflow import DAG
from airflow.utils.dates import days_ago
from model_function import *

start_date = days_ago(1)

default_args = {
    'owner': 'Jose_Oton_',
    'depends_on_past': False,
    'email_on_retry': False,
}

with DAG(
        dag_id='model_pipeline',
        schedule_interval=None,
        tags=['test'],
        description='Test to run bash operator',
        start_date=start_date,
        default_args=default_args,
) as dag:
    csv_create_categorical_variables = build_python_operator(dag=dag, task_name='create_categorical_variables',
                                                             function_name=csv_create_categorical_variables)

    transform_data = build_python_operator(dag=dag, task_name='transform_data', function_name=transform_data)

    delete_cols = build_python_operator(dag=dag, task_name='delete_cols', function_name=delete_cols)

    train_model = build_python_operator(dag=dag, task_name='train_model', function_name=train_model)

    write_results = build_python_operator(dag=dag, task_name='write_results', function_name=write_results)

    csv_create_categorical_variables >> transform_data >> delete_cols >> train_model >> write_results
