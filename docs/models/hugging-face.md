# Hugging Face

AlphAI includes many useful integrations with Hugging Face's open source libraries and AI community. Check out what you can do below


```python
import os
from alphai import AlphAI

aai = AlphAI()
```

## Estimate Memory Requirement

Fun fact: the number of parameters you see in a the name of the model i.e. 3B, 7B, 40B, can easily be converted into a rough estimate of the minimum GB required for your GPU VRAM. Just mulitply by the number by 2 if the parameter dtype is float16 or by 4 if float32!

We've got a fun little helper function to estimate it without even loading the model.


```python
aai.estimate_memory_requirement(model_name="stabilityai/stablelm-zephyr-3b")
```

## Run Generate

You can run local LLMs on GPUs, and we integrate with many open source Hugging Face models.


```python
output = aai.generate("I need a python function that generates the fibonacci sequence recursively.")
```

## Memory Requirement

Now you can get a fairly more accurate size of the model and memory requirement now that the model is loaded.


```python
aai.memory_requirement(model_name_or_path="stabilityai/stablelm-zephyr-3b")
```


```python

```
