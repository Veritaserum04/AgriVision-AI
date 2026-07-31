import gradio as gr
from PIL import Image

from predict import predict_image

def classify(image):

    disease, confidence = predict_image(image)

    return f"""
## 🌿 Prediction

**Disease:** {disease}

**Confidence:** {confidence:.2f}%
"""

demo = gr.Interface(

    fn=classify,

    inputs=gr.Image(type="pil"),

    outputs="markdown",

    title="🌱 AgriVision AI",

    description="Upload a leaf image and detect plant diseases."
)

demo.launch()