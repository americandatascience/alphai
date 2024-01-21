# Jupyter Lab Servers

AlphAI integrates directly with American Data Science's remote Jupyter Lab servers.

You can start servers, stop servers, upload files to your remote file system, load GPU profiling data, and even run code programmatically!

Using these tools require [authentication](authentication.md).

## American Data Science Client

Let's instantiate the `AlphAI` object to profile our tensor run with PyTorch.


```python
import os
from dotenv import load_dotenv

# Make a `.env` file that contains the following line
# ALPHAI_API_KEY=<your-api-key>


load_dotenv()
```


```python
import os
from alphai import AlphAI

aai = AlphAI(
    # Don't need this line if you ran load_dotenv()
    api_key=os.environ.get("ALPHAI_API_KEY"),
)
```

## Start your remote Jupyter Lab servers

Starting up your servers with AlphAI will use the configurations and environment runtime that were used the last time you started the server from your [Dashboard](https://dashboard.amdatascience.com).

If it's a new server or you've never started it, it will use the default AI environment runtime.


```python
aai.start_server()

# Specify the server name if you'd like
#aa.start_server(server_name="Experiment 3")
```


```python
aai.stop_server()
```

## Run code remotely

Alphai allows you to run and "deploy" your code given a string or file path. Your server will automatically start a kernel and run your code remotely.

All servers will also run a tunnel on port 5000, so you could even check out your running servers and apps hosted directly from the server!




```python
hf_code = """from transformers import pipeline
import gradio as gr
gr.close_all()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
demo = gr.Interface.from_pipeline(pipe)
demo.launch(server_port=5000, inline=False)"""

aai.run_code(hf_code, clear_other_kernels=True)
```


```python
aai.get_service()
```

# Upload local files to server


```python
aai.upload(file_path="./test.py")
```
