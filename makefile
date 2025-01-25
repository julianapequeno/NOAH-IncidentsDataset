SHELL := /bin/bash

install:
	cd IncidentsDataset && \
	ls && \
	chmod +x Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
	bash ./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p $(HOME)/miniconda && \
	export PATH="$(HOME)/miniconda/bin:$$PATH" && \
	$(HOME)/miniconda/bin/conda init bash && \
	$(HOME)/miniconda/bin/conda create -n incidents python=3.8.2 -y && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install --upgrade pip && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install googledrivedownloader==0.4 && \
	echo "Current directory: $(shell pwd)" && \
	$(HOME)/miniconda/bin/conda run -n incidents pip install -r $(shell pwd)/IncidentsDataset/requirements.txt
	$(HOME)/miniconda/bin/conda run -n incidents pip install scikit-learn  torch opencv-python matplotlib tqdm ipython-genutils gradio torchvision

train:
	echo "Current directory: $(shell pwd)" && \
	$(HOME)/miniconda/bin/conda run -n incidents python $(shell pwd)/IncidentsDataset/run_download_weights.py && \
	$(HOME)/miniconda/bin/conda run -n incidents python $(shell pwd)/train.py

eval:
	echo "$(shell pwd)" && \
	echo "## Model Metrics" > report.md && \
	cat $(shell pwd)/Results/metrics.txt >> report.md && \
	cml comment create report.md

update-branch:
	cd ./ && \
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git status
	git add -A  # Adiciona arquivos rastreados e não rastreados
	git commit -am "feat: Automatic Update with untracked files"
	git push --force origin HEAD:main

# Login no Hugging Face
hf-login:
	git pull origin main  # Altere para a branch correta, se necessário
	pip install -U "huggingface_hub[cli]"
	echo $(HF) | huggingface-cli login --token $(HF) --add-to-git-credential


push-hub:
	chmod +x $(shell pwd)/setup.sh
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier $(shell pwd)/IncidentsDataset /IncidentsDataset --repo-type=space --commit-message="Sync IncidentsDataset files"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier $(shell pwd)/App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier $(shell pwd)/IncidentsDataset/Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier $(shell pwd)/Results /Metrics --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload julianasky/NOAH-IncidentsDatasetClassifier $(shell pwd)/train.py --repo-type=space --commit-message="Sync train.py"
	

deploy: hf-login push-hub
