# Benchmarking

AlphAI provides a very pythonic and simple approach to time and benchmark your code and callables (functions).

Using these tools do not require authentication.

## Timing

Let's instantiate the `AlphAI` object and create a python function to time and benchmark:


```python
import os
from alphai import AlphAI

aai = AlphAI()
```


```python
import time

# Sleep for x+y seconds
def add_sleep(x, y):
    time.sleep(x+y)
```

Now let's see how long it takes to run this function. It should take around 0.055 seconds; simple!


```python
x = 0.05
y = 0.005

aai.start_timer()
add_sleep(x, y)
aai.stop_timer()
```

## Benchmark

Our `benchmark()` is currently really a timer, but evaluation by comparison is a simple step. Also note that `benchmark` works with key word arguments as well.


```python
x_ = 0.01
y_ = 0.001
```


```python
aai.benchmark(add_sleep, x, y, num_iter=100)
```


```python
aai.benchmark(add_sleep, x_, y_, num_iter=100)
```

### Key Word Arguments


```python
# Sleep for x+y seconds
def add_sleep_kw(x, y = 0.005):
    time.sleep(x+y)
```


```python
aai.benchmark(add_sleep_kw, x_, y=y_, num_iter=100)
```

## Example with PyTorch

Why don't we try this on a forward pass of a PyTorch model!


```python
aai.start_timer()
output = aai.generate("Hello!")
aai.stop_timer()
```


```python
aai.benchmark(aai.generate, "Hello!", num_iter = 5)
```
