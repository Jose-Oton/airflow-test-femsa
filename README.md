# airflow-test-femsa
#### You have to follow the commands below:
- mkdir -p ./dags ./logs ./plugins ./includes
- echo -e "AIRFLOW_UID=$(id -u)" > .env
- docker-compose up airflow-init
- docker-compose up &
- Finally in the Airflow UI (localhost:8080), you have to unpause the dag **model_pipeline** and click **run the dag**
