# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc

all: install clean

test:
	@pytest -v tests

#----------gcloud stuff----------

PROJECT_ID=audio-projects-363306
BUCKET_NAME=buzzfinder_audio_data
REGION=us-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

upload_audio_data:
	@gsutil -m cp -rn audio gs://${BUCKET_NAME}

download_audio_data:
	@gsutil -m cp -rn gs://${BUCKET_NAME}/audio .

#----------mlflow stuff----------

mlflow_launch_tracking_server:
	@mlflow server --backend-store-uri mlruns

EXP_NAME ?= $(shell bash -c 'read -p "Experiment name: "  input; echo $$input')

mlflow_create_experiment:
	@clear
	@mlflow experiments create -n, --experiment-name ${EXP_NAME}

#----------deployment stuff----------

uwsgi_run_server:
	@uwsgi --http 127.0.0.1:5050 --wsgi-file buzzfinder/server.py --callable app --processes 1 --threads 1


docker_compose:
	@docker-compose build
	@docker-compose up
