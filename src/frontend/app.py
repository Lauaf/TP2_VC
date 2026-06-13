from __future__ import annotations

import gradio as gr
from fastapi import FastAPI

from frontend.gradio_ui import build_ui

app = FastAPI(title="Dog Breed Recognition UI", version="0.1.0")
app = gr.mount_gradio_app(app, build_ui(), path="/")
