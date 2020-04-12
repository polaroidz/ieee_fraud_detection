
from datetime import timedelta

from airflow import DAG

from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

from airflow.utils.dates import days_ago

# ==========================================================================
# CODE GOES HERE
# ==========================================================================

def ingest_raw_data(conf, **kwargs):
    print(conf)
    return "It Works"


# ==========================================================================
# DAG GOES HERE
# ==========================================================================
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'f_ingest_raw_data',
    default_args=default_args,
    description='f_ingest_raw_data',
    schedule_interval=timedelta(days=1)
)

t_start = BashOperator(
    task_id='print_date_start',
    bash_command='date',
    dag=dag,
)

t2 = PythonOperator(
    task_id='do_ingest',
    provide_context=True,
    python_callable=ingest_raw_data,
    dag=dag,
)

t_end = BashOperator(
    task_id='print_date_end',
    bash_command='date',
    dag=dag,
)

t_start >> t2 
t2 >> t_end
