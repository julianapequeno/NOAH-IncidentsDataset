PIPELINE_PATH = "/home/runner/work/NOAH-IncidentsDataset/NOAH-IncidentsDataset/IncidentsDataset/Model/incidents_model.joblib"
GH_PATH = "/home/runner/work/NOAH-IncidentsDataset/NOAH-IncidentsDataset"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import torch
import os
from tqdm import tqdm
import joblib
import gradio as gr
import numpy as np
from PIL import Image

# Classe: Carregar imagens
class LoadImages(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, list) and len(X) == 1:
            image_filenames = X
        else:
            raise ValueError("Input must be a list with a single image path.")
        return {"image_filenames": image_filenames}  # Retorna um dicionário

# Classe: Carregar modelo
class LoadModel(BaseEstimator, TransformerMixin):
    def __init__(self, config_filename, checkpoint_folder):
        self.config_filename = config_filename
        self.checkpoint_folder = checkpoint_folder
    
    def fit(self, X=None, y=None):
        from IncidentsDataset.architectures import (
            get_incidents_model,
            update_incidents_model_with_checkpoint,
            update_incidents_model_to_eval_mode
        )
        from IncidentsDataset.parser import get_parser, get_postprocessed_args

        parser = get_parser()
        args = parser.parse_args(
            args=f"--config={self.config_filename} --checkpoint_path={self.checkpoint_folder} --mode=test --num_gpus=0"
        )
        self.args = get_postprocessed_args(args)
        self.model = get_incidents_model(self.args)
        update_incidents_model_with_checkpoint(self.model, self.args)
        update_incidents_model_to_eval_mode(self.model)
        return self
    
    def transform(self, X):
        return {"image_filenames": X['image_filenames'],"model": self.model, "args": self.args}

# Classe: Executar inferência
class RunInference(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=1, num_workers=4, topk=5):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.topk = topk
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X):
        image_filenames = X["image_filenames"]
        model = X["model"]
        args = X["args"]

        from IncidentsDataset.architectures import get_predictions_from_model, FilenameDataset
        from IncidentsDataset.utils import get_index_to_incident_mapping, get_index_to_place_mapping

        dataset = FilenameDataset(image_filenames, targets=image_filenames)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        inference_dict = {}
        for _, (batch_input, image_paths) in tqdm(enumerate(loader)):
            get_predictions_from_model(
                args,
                model,
                batch_input,
                image_paths,
                get_index_to_incident_mapping(),
                get_index_to_place_mapping(),
                inference_dict,
                topk=self.topk
            )
        return inference_dict

# Classe: Combinar passos
class CombineSteps(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, data):
        return {
            "image_filenames": data["image_filenames"],
            "model": data["model"],
            "args": data["args"]
        }

# Configurações
CONFIG_FILENAME = f"{GH_PATH}/IncidentsDataset/configs/eccv_final_model"
CHECKPOINT_PATH_FOLDER = f"{GH_PATH}/IncidentsDataset/pretrained_weights/"

# Pipeline ajustado
pipeline = Pipeline([
    ("load_images", LoadImages()),
    ("load_model", LoadModel(CONFIG_FILENAME, CHECKPOINT_PATH_FOLDER)),
    ("combine", CombineSteps()),
    ("run_inference", RunInference())
])

if not os.path.exists(f"{GH_PATH}/IncidentsDataset/Model/"):
  os.mkdir(f"{GH_PATH}/IncidentsDataset/Model/")

# Salvar o pipeline
MODEL_PATH = f"{GH_PATH}/IncidentsDataset/Model/incidents_model.joblib"
joblib.dump(pipeline, MODEL_PATH)

######## Movido papra App/incidents_app.py
# Carregar o modelo
#pipeline = joblib.load(MODEL_PATH)
#
# Função Gradio
#def classify_image(image):
#   # Converter imagem para formato esperado
#    image = np.array(Image.fromarray(image).convert("RGB"))
#    
#    # Salvar imagem temporária
#    temp_image_path = "temp_image.jpg"
#   cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#    
#    try:
#        # Passar caminho da imagem para o pipeline
#        inference_results = pipeline.fit_transform([temp_image_path])
#        predictions = inference_results[temp_image_path]
#       # with open("/content/CI-CD-IncidentsModel/Results/metrics.txt", "w") as outfile:
#         # outfile.write(f"\n[{str(image_filename)} ] Os incidentes: {predictions['incidents']} \t Probabilidades: {predictions['incident_probs']} \n Os lugares: {predictions['places']} \t Probabilidades: {predictions['place_probs']}")
#
#        
#       # Formatar saída
#        result = {
#            "Incidentes": predictions["incidents"],
#            "Probabilidades dos Incidentes": predictions["incident_probs"],
#            "Lugares": predictions["places"],
#            "Probabilidades dos Lugares": predictions["place_probs"],
#        }
#    except Exception as e:
#        result = {"Erro": str(e)}
#    
#    return result
#
# Interface Gradio
#demo = gr.Interface(
#    fn=classify_image,
#    inputs=gr.Image(type="numpy", label="Envie uma Imagem"),
#    outputs=gr.JSON(label="Predições"),
#    title="Classificador de Incidentes",
#    description="Envie uma imagem para classificar os incidentes e lugares associados.",
#)
#
#if __name__ == "__main__":
#    demo.launch(share=True)
