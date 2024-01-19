from transformers import pipeline
import gradio as gr
gr.close_all()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
demo = gr.Interface.from_pipeline(pipe)
demo.launch(server_port=5000, inline=False)