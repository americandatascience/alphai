# Authentication 

## Setup and Pre-requisites

Although not strictly required to use the computational functions of the alphai package, it is recommended to create an account at [American Data Science](https://dashboard.amdatascience.com) and generate an API key to make use of your two free remote Jupyter Lab servers.

You don't need an API key to use the GPU profiling, benchmarking, and generate modules.

### Get your API key

You can generate your API key after logging in and visiting your profile page [here](https://dashboard.amdatascience.com/profile).


### Try it out

If you're authenticated with your AmDS account and api key, you can start and stop your servers.


```python
import os
from alphai import AlphAI

aai = AlphAI(
    api_key=os.environ.get("ALPHAI_API_KEY"),
)

# Starting default server
# May take a moment to get ready
aai.start_server()

# Upload to your server's file system 
aai.upload("./main.py")

# Start python kernel and run code remotely
code = "print('Hello world!')"
aai.run_code(code)

```


To stop the servers run below:


```python
aai.stop_server()
```

