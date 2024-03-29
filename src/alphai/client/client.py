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

        # Initialize server
        self.initialize_server()

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
    def start_server(
            self,
            server_name: str = "",
            environment: str = "ai",
            server_request: str = "medium-cpu",
        ):

        # Start Server given name
        # Data to be sent in POST request
        data = {
            "server_name": server_name,
            "environment": environment,
            "server_request": server_request,
            "port": 5000,
        }

        url = f"{self.dashboard_url}/api/server"
        headers = {
            "apikey": f"{self.access_token}",
            'Accept': 'application/json'
        }

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


    def stop_server(self, server_name: str = ""):
        # Stop Server given name
        # Data to be sent in POST request
        if not server_name:
            server_name = "default"
        data = {
            "stop": True,
        }

        url = f"{self.dashboard_url}/api/server/{server_name}"
        headers = {
            "apikey": f"{self.access_token}",
            'Accept': 'application/json'
            }

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

    def alph(
            self,
            server_name: str = "",
            messages: str | list = "Hi Alph.",
            engine: str = "gpt3",
        ):

        # Agent Alph call
        # Data to be sent in POST request
        if isinstance(messages, str):
            data = {
                "messages": [
                    {"role": "user", "content": messages}
                ],
            }
        else:
            data = {
                "messages": messages
            }

        url = f"{self.dashboard_url}/api/alph/{server_name}/{engine}"
        headers = {
            "apikey": f"{self.access_token}",
            'Accept': 'application/json'
        }

        response = requests.post(
            url,
            json=data,
            headers=headers,
            stream=True
        )

        # If the response is JSON
        #try:
        if response.encoding is None:
            response.encoding = 'utf-8'

        full_output = []
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    #import pdb; pdb.set_trace()
                    split_line = line.split(':')
                    cleaned_line = split_line[1].replace('"', '')
                    print(cleaned_line, end="")
                    full_output.append(cleaned_line)
        except ValueError:
            print("Response encoded incorrectly.")
        
        return "".join(full_output)

    # NB server client
    def get_contents(self, server_name: str = ""):
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
        # Get initial kernel info
        url = f"{self.host}/user/{self.user_name}/{server_name}/api/kernels"
        headers = {"Authorization": f"Token {self.access_token}"}
        response = requests.get(url, headers=headers)
        kernels = json.loads(response.text)
        self.kernels = kernels
        return kernels

    def shutdown_all_kernels(self, server_name=""):
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

    def get_service(self, server_name: str = ""):
        server_name = f"--{server_name}" if server_name else ""
        user_name = self.user_name.replace("@", "-40").replace(".", "-2e")
        url = f"https://jupyter-{user_name}{server_name}.americandatascience.dev"
        return url
