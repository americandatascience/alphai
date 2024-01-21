# Profiling GPU

AlphAI provides a simple and straight forward profiling process to analyze your tensor processes on your GPUs.

Using these tools do not require authentication. However, authentication is required for `load_view()`.

## Profiling with PyTorch

Let's instantiate the `AlphAI` object to profile our tensor run with PyTorch.


```python
import os
from alphai import AlphAI

aai = AlphAI()
```

### With start() and stop()


```python
import torch
import math

aai.start()
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
x = torch.linspace(-math.pi, math.pi, 2000)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
aai.stop()


```

### With context manager


```python
with aai:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    x = torch.linspace(-math.pi, math.pi, 2000)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

```

### Run profiler analytics


```python
aai.run_profiler_analysis()
```


```python
aai.get_averages()
```

### Save


```python
aai.save()
```

## Profile and Load View

If you'd like to run `load_view()` and to see your GPU usage statistics and more, you'll need to authenticate.


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

Now just wrap any PyTorch GPU operation you'd like to profile.

```python
import math
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
).to("cuda")
x = torch.linspace(-math.pi, math.pi, 2000)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p).to("cuda")

aai.start()
model(xx)
aai.stop()
```

Run profiler analytics and save your trace and analytics locally.

```
aai.run_profiler_analysis()
aai.save()
```

Then run load view to use the online [viewer](https://amdatascience.com/viewer) for your analytics and statistics.


```python
aai.load_view()
```
