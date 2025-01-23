
PIPELINE_PATH = "./IncidentsDataset/Model/incidents_model.joblib"
GH_PATH = "."

import cv2
import os
from tqdm import tqdm
import joblib
import gradio as gr
import numpy as np
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import train   # Acesso à função define no train.py

MODEL_PATH = f"{GH_PATH}/IncidentsDataset/Model/incidents_model.joblib"
# Carregar o modelo
pipeline = joblib.load(MODEL_PATH)

# Função Gradio
def classify_image(image):
    # Converter imagem para formato esperado
    image = np.array(Image.fromarray(image).convert("RGB"))
    
    # Salvar imagem temporária
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    try:
        # Passar caminho da imagem para o pipeline
        inference_results = pipeline.fit_transform([temp_image_path])
        predictions = inference_results[temp_image_path]
       # with open("/content/CI-CD-IncidentsModel/Results/metrics.txt", "w") as outfile:
         # outfile.write(f"\n[{str(image_filename)} ] Os incidentes: {predictions['incidents']} \t Probabilidades: {predictions['incident_probs']} \n Os lugares: {predictions['places']} \t Probabilidades: {predictions['place_probs']}")

        
        # Formatar saída
        result = {
            "Incidentes": predictions["incidents"],
            "Probabilidades dos Incidentes": predictions["incident_probs"],
            "Lugares": predictions["places"],
            "Probabilidades dos Lugares": predictions["place_probs"],
        }
    except Exception as e:
        result = {"Erro": str(e)}
    
    return result

# Interface Gradio
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Envie uma Imagem"),
    outputs=gr.JSON(label="Predições"),
    title="Classificador de Incidentes",
    description="Envie uma imagem para classificar os incidentes e lugares associados.",
)

if __name__ == "__main__":
    demo.launch(share=True)