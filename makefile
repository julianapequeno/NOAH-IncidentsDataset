SHELL := /bin/bash

install:
	cd IncidentsDataset && \
	ls && \
	wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
	chmod +x Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
	bash ./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p $(HOME)/miniconda && \
	export PATH="$(HOME)/miniconda/bin:$$PATH" && \
	$(HOME)/miniconda/bin/conda init bash && \
	$(HOME)/miniconda/bin/conda create -n incidents python=3.8.2 -y && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install --upgrade pip && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install googledrivedownloader==0.4 && \
	echo "Current directory: $(shell pwd)" && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install -r /IncidentsDataset/requirements.txt
	$(HOME)/miniconda/bin/conda run -n incidents pip install scikit-learn  torch opencv-python matplotlib tqdm ipython-genutils gradio

train:
	echo "Current directory: $(shell pwd)" && \
	$(HOME)/miniconda/bin/conda run -n incidents python ./IncidentsDataset/run_download_weights.py && \
	$(HOME)/miniconda/bin/conda run -n incidents python ./train.py

eval:
	echo "## Model Metrics" > report.md && \
	cat /Results/metrics.txt >> report.md && \
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .  # Adiciona os arquivos não rastreados
	git commit -am "Update with new results"
	git push --force origin HEAD:main

# Login no Hugging Face
hf-login:
	git pull origin main  # Altere para a branch correta, se necessário
	pip install -U "huggingface_hub[cli]"
	echo $(HF) | huggingface-cli login --token $(HF) --add-to-git-credential


push-hub:
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier ./IncidentsDataset /IncidentsDataset --repo-type=space --commit-message="Sync IncidentsDataset files"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier ./IncidentsDataset/Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub
