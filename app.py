import gradio as gr

from predict import predict_image
from disease_info import DISEASE_INFO


def classify(image):

    disease, confidence = predict_image(image)

    info = DISEASE_INFO.get(
        disease,
        {
            "description": "No description available.",
            "treatment": "No treatment available."
        }
    )

    return f"""
# 🌿 AgriVision AI

## Disease
**{disease}**

## 🎯 Confidence
**{confidence:.2f}%**

## 🦠 Description
{info['description']}

## 💊 Recommended Treatment
{info['treatment']}
"""


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="markdown",
    title="🌱 AgriVision AI",
    description="Upload a leaf image to detect plant diseases."
)

demo.launch()