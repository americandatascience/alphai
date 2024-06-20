# src/alphai/client/client.py
import requests
from typing import List
import urllib.parse
import urllib.request
import base64
import json
from websocket import create_connection, WebSocketTimeoutException
import uuid
import datetime
import importlib
from platform import platform

import nbserv_client
import os
import jh_client
from pprint import pprint


class Client:
    def __init__(
        self,
        host: str = "https://lab.amdatascience.com",
        dashboard_url: str = "https://dashboard.amdatascience.com",
        access_token=None,
        timeout: float = None,
        headers: dict = {},
    ):
        self.host = host
        self.dashboard_url = dashboard_url
        self.access_token = access_token
        self.configuration = jh_client.Configuration(host=f"{host}/hub/api")
        self.configuration.access_token = access_token

        # Enter a context with an instance of the API client
        self.api_client = jh_client.ApiClient(self.configuration)
        # Create an instance of the API class
        self.api_instance = jh_client.DefaultApi(self.api_client)

        # Headers
        self.headers = headers
        self.headers["User-Agent"] = (
            f"alphai sdk v-{get_alphai_version()}_{platform().lower()}"
        )
        self.headers["X-Device-Id"] = str(uuid.UUID(int=uuid.getnode()))
        self.headers["X-metrics"] = "{}"
        self.headers.update(
            {"apikey": f"{self.access_token}", "Accept": "application/json"}
        )
        self.timeout = timeout

        # Initialize server
        self.initialize_server()

    def _request(self, method, endpoint, data=None):
        url = f"{self.dashboard_url}/{endpoint}"
        response = requests.request(method, url, json=data, headers=self.headers)
        try:
            return response.json()
        except ValueError:
            print("Response is not in JSON format")

    def get(self, endpoint):
        return self._request("GET", endpoint)

    def post(self, endpoint, data):
        return self._request("POST", endpoint, data)

    def put(self, endpoint, data):
        return self._request("PUT", endpoint, data)

    def delete(self, endpoint):
        return self._request("DELETE", endpoint)

    def initialize_server(self, server_name=""):
        self.server_name = server_name
        try:
            # Return authenticated user's model
            self.user_api_response = self.api_instance.user_get()
            self.user_info = self.user_api_response.to_dict()

        except Exception as e:
            print("Exception when calling DefaultApi->user_get: %s\n" % e)

        self.user_name = self.user_info["name"]
        self.server_configuration = nbserv_client.Configuration(
            host=f"{self.host}/user/{self.user_name}/{server_name}"
        )
        self.server_api_client = nbserv_client.ApiClient(
            self.server_configuration,
            header_name="Authorization",
            header_value=f"Token {self.access_token}",
        )
        self.server_api_instance = nbserv_client.ContentsApi(self.server_api_client)

    def get_user_info(self):
        return self.user_info

    def get_servers(self):
        user_api_response = self.api_instance.users_name_get(
            self.user_name, include_stopped_servers=True
        )
        servers = user_api_response.to_dict()["servers"]
        return {
            k: {
                "name": v["name"],
                "ready": v["ready"],
                "url": v["url"],
                "last_activity": v["last_activity"],
            }
            for k, v in servers.items()
        }

    # Dashboard Client
    def servers(
        self,
    ):
        # Get Servers
        response = self.get(
            endpoint="api/server",
        )

        return response

    def create_server(
        self,
        server_name: str = "default",
        environment: str = "ai",
        compute: str = "amds-medium_cpu",
        port: int = 5000,
    ):

        # Start Server given name
        # Data to be sent in POST request
        if not server_name:
            server_name = "default"
        data = {
            "server_name": server_name,
            "environment": environment,
            "server_request": compute,
            "port": port,
        }

        response = self.post(
            endpoint="api/server",
            data=data,
        )

        return response

    def start_server(
        self,
        server_name: str = "default",
        environment: str = "ai",
        compute: str = "amds-medium_cpu",
        port: int = 5000,
    ):

        # Start Server given name
        # Data to be sent in POST request
        if not server_name:
            server_name = "default"
        data = {
            "environment": environment,
            "server_request": compute,
            "port": port,
        }

        response = self.post(
            endpoint=f"api/server/{server_name}",
            data=data,
        )

        return response

    def stop_server(self, server_name: str = "default"):
        # Stop Server given name
        # Data to be sent in POST request
        if not server_name:
            server_name = "default"
        data = {
            "stop": True,
        }

        response = self.post(
            endpoint=f"api/server/{server_name}",
            data=data,
        )

        return response

    def delete_server(self, server_name: str = "default"):
        # Delete Server given name
        # Data to be sent in POST request
        if not server_name:
            server_name = "default"

        response = self.delete(
            endpoint=f"api/server/{server_name}",
        )

        return response

    def alph(
        self,
        server_name: str = "default",
        messages: list = [{"role": "user", "content": "Hi Alph!"}],
        engine: str = "gpt-3",
    ):
        # Agent Alph call
        if not server_name:
            server_name = "default"

        # Data to be sent in POST request
        data = {"messages": messages}

        url = f"{self.dashboard_url}/api/alph/{server_name}/{engine}"

        response = requests.post(url, json=data, headers=self.headers, stream=True)

        # If the response is JSON
        # try:
        if response.encoding is None:
            response.encoding = "utf-8"

        full_output = []
        # Decode and stream text properly
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # import pdb; pdb.set_trace()
                    split_line = line.split(":", 1)
                    try:
                        # import pdb; pdb.set_trace()
                        tool_call = json.loads(split_line[1])
                        cleaned_line = json.dumps(tool_call.get("result", ""), indent=4)
                        print("Analyzing...")
                        print(cleaned_line)
                    except:
                        cleaned_line = split_line[1].replace('"', "")
                        if r"\n" in cleaned_line:
                            print(cleaned_line.replace(r"\n", ""))
                        else:
                            print(cleaned_line, end="")
                    full_output.append(cleaned_line)
        except ValueError:
            print("Response encoded incorrectly.")

        return "".join(full_output)

    # NB Server Client

    def get_contents(self, server_name: str = ""):
        if server_name == "default":
            server_name = ""
        if server_name != self.server_name:
            self.initialize_server(server_name=server_name)
        path = ""  # str | file path
        type = None  # str | File type ('file', 'directory') (optional)
        format = "text"  # str | How file content should be returned ('text', 'base64') (optional)
        content = None  # int | Return content (0 for no content, 1 for return content) (optional)
        hash = None  # int | May return hash hexdigest string of content and the hash algorithm (0 for no hash - default, 1 for return hash). It may be ignored by the content manager. (optional)

        try:
            # Get contents of file or directory
            api_response = self.server_api_instance.api_contents_path_get(
                path, type=type, format=format, content=content, hash=hash
            )
            print("The response of ContentsApi->api_contents_path_get:\n")
            pprint(api_response)
        except Exception as e:
            print("Exception when calling ContentsApi->api_contents_path_get: %s\n" % e)

    def post_contents(
        self,
        server_name: str = "",
        path: str = "",
        ext: str = "",
        type: str = "directory",
    ):
        if server_name == "default":
            server_name = ""

        # Data to be sent in POST request
        data = {
            "type": type,
            "ext": ext,
        }

        url = f"{self.host}/user/{self.user_name}/{server_name}/api/contents/{path}"
        headers = {"Authorization": f"Token {self.access_token}"}

        response = requests.post(
            url,
            json=data,
            headers=headers,
        )

        # If the response is JSON
        try:
            response_data = response.json()
            return response_data
        except ValueError:
            print("Response is not in JSON format")

    def patch_contents(
        self,
        server_name: str = "",
        path: str = "Untitled Folder",
        new_path: str = "alphai_",
    ):
        if server_name == "default":
            server_name = ""
        # Data to be sent in PATCH request
        data = {"path": new_path}

        url = f"{self.host}/user/{self.user_name}/{server_name}/api/contents/{path}"
        headers = {"Authorization": f"Token {self.access_token}"}

        response = requests.patch(
            url,
            json=data,
            headers=headers,
        )

        # If the response is JSON
        try:
            response_data = response.json()
            return response_data
        except ValueError:
            print("Response is not in JSON format")

    def put_contents(self, server_name: str = "", path: str = "", file_path: str = ""):
        if server_name == "default":
            server_name = ""
        # Data to be sent in POST request
        file_name = file_path[1 + file_path.rfind(os.sep) :]
        url = f"{self.host}/user/{self.user_name}/{server_name}/api/contents/{path}/{file_name}"
        headers = {"Authorization": f"Token {self.access_token}"}

        try:
            with open(file_path, "rb") as f:
                data = f.read()
                b64data = base64.b64encode(data)
                body = json.dumps(
                    {
                        "content": b64data.decode(),
                        "name": file_name,
                        "path": path,
                        "format": "base64",
                        "type": "file",
                    }
                )
                return requests.put(url, data=body, headers=headers, verify=True)
        except ValueError:
            print("Request is invalid")

    def get_kernels(self, server_name=""):
        if server_name == "default":
            server_name = ""
        # Get initial kernel info
        url = f"{self.host}/user/{self.user_name}/{server_name}/api/kernels"
        headers = {"Authorization": f"Token {self.access_token}"}
        response = requests.get(url, headers=headers)
        kernels = json.loads(response.text)
        self.kernels = kernels
        return kernels

    def shutdown_all_kernels(self, server_name=""):
        if server_name == "default":
            server_name = ""
        # Get initial kernel info
        kernels = self.get_kernels(server_name=server_name)
        # Delete all kernels
        headers = {"Authorization": f"Token {self.access_token}"}
        for k in kernels:
            url = (
                f"{self.host}/user/{self.user_name}/{server_name}/api/kernels/{k['id']}"
            )
            response = requests.delete(url, headers=headers)

    def send_channel_execute(
        self,
        server_name="",
        messages: List[str] = ["print('Hello World!')"],
        return_full: bool = False,
    ):
        if server_name == "default":
            server_name = ""
        # start initial kernel info
        url = f"{self.host}/user/{self.user_name}/{server_name}/api/kernels"
        headers = {"Authorization": f"Token {self.access_token}"}
        response = requests.post(url, headers=headers)
        kernel = json.loads(response.text)

        # Execution request/reply is done on websockets channels
        ws_url = f"wss://{self.host.split('https://')[-1]}/user/{self.user_name}/{urllib.parse.quote(server_name)}/api/kernels/{kernel['id']}/channels"
        ws = create_connection(ws_url, header=headers)

        code = messages

        def execute_request(code):
            msg_type = "execute_request"
            content = {"code": code, "silent": False}
            hdr = {
                "msg_id": uuid.uuid1().hex,
                "username": "test",
                "session": uuid.uuid1().hex,
                "data": datetime.datetime.now().isoformat(),
                "msg_type": msg_type,
                "version": "5.0",
            }
            msg = {
                "header": hdr,
                "parent_header": hdr,
                "metadata": {},
                "content": content,
            }
            return msg

        for c in code:
            if not c.startswith("!"):
                c += ";print('AlphAI Run Complete')"
            ws.send(json.dumps(execute_request(c)))

        results = {}
        for i in range(0, len(code)):
            msg_type = ""
            results[code[i]] = []
            count = 0
            while msg_type != "stream":
                try:
                    rsp = json.loads(ws.recv())
                    results[code[i]].append(rsp)
                    # print(rsp["msg_type"])
                    # print(rsp["content"])
                    if rsp["msg_type"] == "stream":
                        print(rsp["content"]["text"])
                    msg_type = rsp["msg_type"]
                    if msg_type == "error":
                        raise Exception(rsp["content"]["traceback"][0])
                except WebSocketTimeoutException as _e:
                    print("No output")
                    return
        ws.close()

        if return_full:
            return results


def get_alphai_version() -> str:
    """Grabs the version of AlphAI from builtin importlib library

    Returns:
        str: version name, unknown if fails to dynamically pull
    """
    try:
        return importlib.metadata.version("alphai")
    except:
        return "unknown"
