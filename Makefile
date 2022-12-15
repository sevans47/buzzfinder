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


#----------gcloud bucket stuff----------

PROJECT_ID=audio-projects-363306
BUCKET_NAME=buzzfinder_audio_data
REGION=us-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

upload_audio_data:
	@gsutil -m cp -rn audio gs://${BUCKET_NAME}

download_audio_data:
	@gsutil -m cp -rn gs://${BUCKET_NAME}/audio .

upload ml_runs:
	@gsutil -m cp -rn mlruns gs://${BUCKET_NAME}

download_ml_runs:
	@gsutil -m cp -rn gs://${BUCKET_NAME}/mlruns .


#----------mlflow stuff----------

mlflow_launch_tracking_server:
	@mlflow server --backend-store-uri mlruns

EXP_NAME ?= $(shell bash -c 'read -p "Experiment name: "  input; echo $$input')

mlflow_create_experiment:
	@clear
	@mlflow experiments create -n, --experiment-name ${EXP_NAME}


#----------deployment----------

# put trained model in "data" folder before running the following bentoml_save_model
MODEL_NAME ?= $(shell bash -c 'read -p "file name for model: "  input; echo $$input')

bentoml_save_model:
	@python buzzfinder/savemodeltobento.py ${MODEL_NAME}

bentoml_view_models:
	@bentoml models list

# after you create service.py and bentofile.yaml
bentoml_build:
	@cd api && bentoml build

bentoml_serve:
	@cd api && bentoml serve buzzfinder_classifier:latest --production

bentoctl_init:
	@cd api && bentoctl init

bentoctl_build_and_push_docker_image:
	@cd api && bentoctl build -b buzzfinder_classifier: latest -f deployment_config.yaml

terraform_init:
	@cd api && terraform init

terraform_deploy:
	@cd api && terraform apply -var-file=bentoctl.tfvars -auto-approve


#----------deployment stuff----------


# docker_compose:
# 	@docker-compose build
# 	@docker-compose up

# DOCKER_IMAGE_NAME=buzzfinder_classifier
# GCR_MULTI_REGION=us.gcr.io

# docker_push:
# 	@docker push ${GCR_MULTI_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

# docker_run:
# 	@docker run -e PORT=8000 -p 8080:8000 ${GCR_MULTI_REGION}/${PROJECT_ID}/${DOCKER_IMAGE_NAME}
