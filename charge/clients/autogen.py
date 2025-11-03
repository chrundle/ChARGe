try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_core.models import ModelFamily, ChatCompletionClient, CreateResult
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_agentchat.conditions import TextMentionTermination
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

from functools import partial
import os
from charge.clients.Client import Client
from typing import Type, Optional, Dict, Union, List, Any
from charge.Experiment import Experiment
from charge.clients.hf import HuggingFaceLocalClient

class AutoGenClient(Client):
    def __init__(
        self,
        experiment_type: Experiment,
        path: str = ".",
        max_retries: int = 3,
        backend: str = "openai",
        model: str = "gpt-4",
        model_client: Optional[ChatCompletionClient] = None,
        api_key: Optional[str] = None,
        model_info: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        server_path: Optional[Union[str, list[str]]] = None,
        server_url: Optional[Union[str, list[str]]] = None,
        server_kwargs: Optional[dict] = None,
        max_tool_calls: int = 15,
        check_response: bool = False,
        max_multi_turns: int = 100,
        # New parameters for local HuggingFace models
        local_model_path: Optional[str] = "/p/vast1/flask/models/gpt-oss-120b",
        device: str = "auto",
        torch_dtype: str = "auto",
        quantization: Optional[str] = "4bit",
    ):
        """Initializes the AutoGenClient.

        Args:
            experiment_type (Type[Experiment]): The experiment class to use.
            path (str, optional): Path to save generated MCP server files. Defaults to ".".
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            backend (str, optional): Backend to use: "openai", "gemini", "ollama", "huggingface", "livai" or "livchat". Defaults to "openai".
            model (str, optional): Model name to use. Defaults to "gpt-4".
            model_client (Optional[ChatCompletionClient], optional): Pre-initialized model client. If provided, `backend`, `model`, and `api_key` are ignored. Defaults to None.
            api_key (Optional[str], optional): API key for the model. Defaults to None.
            model_info (Optional[dict], optional): Additional model info. Defaults to None.
            model_kwargs (Optional[dict], optional): Additional keyword arguments for the model client.
                                                     Defaults to None.
            server_path (Optional[Union[str, list[str]]], optional): Path or list of paths to existing MCP server script. If provided, this
                                                   server will be used instead of generating
                                                   new ones. Defaults to None.
            server_url (Optional[Union[str, list[str]]], optional): URL or list URLs of existing MCP server over the SSE transport.
                                                  If provided, this server will be used instead of generating
                                                  new ones. Defaults to None.
            server_kwargs (Optional[dict], optional): Additional keyword arguments for the server client. Defaults to None.
            max_tool_calls (int, optional): Maximum number of tool calls per task. Defaults to 15.
            check_response (bool, optional): Whether to check the response using verifier methods.
                                             Defaults to False (Will be set to True in the future).
            max_multi_turns (int, optional): Maximum number of multi-turn interactions. Defaults to 100.
            local_model_path (Optional[str], optional): Path to local HuggingFace model directory. 
                                                       Required when backend="huggingface". Defaults to local FLASK gpt-oss-120b.
            device (str, optional): Device to load model on ("auto", "cuda", "cpu"). Defaults to "auto".
            torch_dtype (str, optional): Torch dtype for model ("auto", "float16", "bfloat16"). Defaults to "auto".
            quantization (Optional[str], optional): Quantization method ("4bit", "8bit", None). Defaults to "4bit".
        
        Raises:
            ValueError: If neither `server_path` nor `server_url` is provided and MCP servers cannot be generated.
        """
        super().__init__(experiment_type, path, max_retries)
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.model_info = model_info
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.max_tool_calls = max_tool_calls
        self.check_response = check_response
        self.max_multi_turns = max_multi_turns
        
        # Initialize servers list if not already done by parent
        if not hasattr(self, 'servers'):
            self.servers = []

        if model_client is not None:
            self.model_client = model_client
        else:
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": True,
            }
            
            if backend == "huggingface":
                # Use local HuggingFace model
                if local_model_path is None:
                    raise ValueError(
                        "local_model_path must be provided when backend='huggingface'"
                    )
                
                self.model_client = HuggingFaceLocalClient(
                    model_path=local_model_path,
                    model_info=model_info,
                    device=device,
                    torch_dtype=torch_dtype,
                    quantization=quantization,
                    **self.model_kwargs,
                )
            elif backend == "vllm":
                # Use vLLM server
                vllm_url = self.model_kwargs.get(
                    "vllm_url", 
                    os.getenv("VLLM_URL", "http://localhost:8000/v1")
                )
                vllm_model = self.model_kwargs.get(
                    "vllm_model", 
                    os.getenv("VLLM_MODEL", model or "/p/vast1/flask/models/gpt-oss-120b")
                )
                reasoning_effort = self.model_kwargs.get(
                    "reasoning_effort", 
                    "medium"
                )

                #self.model_kwargs["max_tokens"] = 10000 # Doesn't seem to work
                #print(f"\n  ==> MY model_kwargs: {self.model_kwargs}")
                #print(f"\n  ==> vllm backend vllm_url: {vllm_url}")
                #print(f"\n  ==> vllm backend vllm_model: {vllm_model}")
                print(f"\n  ==> GPT-OSS reasoning effort: {reasoning_effort}")

                from autogen_ext.models.openai import OpenAIChatCompletionClient
                
                vllm_model_info = {
                    "vision": False,
                    "function_calling": True,
                    "json_output": False, #True, # TODO determine best choice
                    "family": ModelFamily.UNKNOWN,
                    "structured_output": False, #True, # TODO determine best choice
                }
                from charge.clients.reasoning import ReasoningCaptureClient
                self.model_client = ReasoningCaptureClient(
                    model=vllm_model,
                    api_key="EMPTY",
                    base_url=vllm_url,
                    model_info=vllm_model_info,
                    parallel_tool_calls=False,
                    extra_body={"reasoning_effort": reasoning_effort},
                )
                #from charge.clients.logging import LoggingModelClient, InspectingModelClient
                #self.model_client = LoggingModelClient(
                #    model=vllm_model,
                #    api_key="EMPTY",
                #    base_url=vllm_url,
                #    model_info=vllm_model_info,
                #    parallel_tool_calls=False,
                #    extra_body={"reasoning_effort": reasoning_effort},
                #)
                #self.model_client = OpenAIChatCompletionClient(
                #    model=vllm_model,
                #    api_key="EMPTY",
                #    base_url=vllm_url,
                #    model_info=vllm_model_info,
                #    parallel_tool_calls=False, # Seems more reliable when False
                #    extra_body={"reasoning_effort": reasoning_effort},
                #)

                #from charge.clients.vllm import VLLMClient, VLLMOpenAIChatClient
                #self.model_client = VLLMOpenAIChatClient(
                #    model=vllm_model,
                #    api_key="EMPTY",
                #    base_url=vllm_url,
                #    model_info=vllm_model_info,
                #    parallel_tool_calls=False, # Seems more reliable when False
                #    extra_body={"reasoning_effort": reasoning_effort},
                #)
                
                #self.model_client = VLLMClient(
                #    base_url=vllm_url,
                #    model_name=vllm_model,
                #    model_info=model_info,
                #    reasoning_effort=reasoning_effort,
                #)
            elif backend == "ollama":
                from autogen_ext.models.ollama import OllamaChatCompletionClient

                self.model_client = OllamaChatCompletionClient(
                    model=model,
                    model_info=model_info,
                )
            else:
                from autogen_ext.models.openai import OpenAIChatCompletionClient

                if api_key is None:
                    if backend == "gemini":
                        api_key = os.getenv("GOOGLE_API_KEY")
                    else:
                        api_key = os.getenv("OPENAI_API_KEY")
                assert (
                    api_key is not None
                ), "API key must be provided for OpenAI or Gemini backend"
                self.model_client = OpenAIChatCompletionClient(
                    model=model,
                    api_key=api_key,
                    model_info=model_info,
                    **self.model_kwargs,
                )

        if server_path is None and server_url is None:
            self.setup_mcp_servers()
        else:
            if server_path is not None:
                if isinstance(server_path, str):
                    server_path = [server_path]
                for sp in server_path:
                    self.servers.append(StdioServerParams(command="python3", args=[sp]))
            if server_url is not None:
                if isinstance(server_url, str):
                    server_url = [server_url]
                for su in server_url:
                    self.servers.append(
                        SseServerParams(url=su, **(server_kwargs or {}))
                    )
        self.messages = []

    @staticmethod
    def configure(model: Optional[str], backend: str) -> tuple[str, str, Optional[str], Dict[str, str]]:
        import httpx

        kwargs = {}
        API_KEY = None
        default_model = None
        if backend in ["openai", "gemini", "livai", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                default_model = "gpt-4"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
            elif backend == "livai" or backend == "livchat":
                API_KEY = os.getenv("OPENAI_API_KEY")
                BASE_URL = os.getenv("LIVAI_BASE_URL")
                assert (
                    BASE_URL is not None
                ), "LivAI Base URL must be set in environment variable"
                default_model = "gpt-4.1"
                kwargs["base_url"] = BASE_URL
                kwargs["http_client"] = httpx.AsyncClient(verify=False)
            else:
                API_KEY = os.getenv("GOOGLE_API_KEY")
                default_model = "gemini-flash-latest"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
        elif backend in ["ollama"]:
            default_model = "gpt-oss:latest"
        elif backend in ["huggingface"]:
            default_model = None  # Must be provided via local_model_path
        elif backend in ["vllm"]:
            kwargs["reasoning_effort"] = os.getenv("OSS_REASONING", "medium")
            default_model = "gpt-oss"  # Default vLLM model name

        if not model:
            model = default_model
        return (model, backend, API_KEY, kwargs)

    def check_invalid_response(self, result) -> bool:
        answer_invalid = False
        for method in self.verifier_methods:
            try:
                is_valid = method(result.messages[-1].content)
                if not is_valid:
                    answer_invalid = True
                    break
            except Exception as e:
                print(f"Error during verification with {method.__name__}: {e}")
                answer_invalid = True
                break
        return answer_invalid

    async def step(self, agent, task: str):
        result = await agent.run(task=task)

        # Debug: Check for tool calls in the result
        ##print(f"\n=== DEBUG: Step Result ===")
        ##print(f"Number of messages: {len(result.messages)}")
        ##for i, msg in enumerate(result.messages):
        ##    print(f"Message {i}: type={type(msg).__name__}")
        ##    if hasattr(msg, 'content'):
        ##        print(f"  Content: {str(msg.content)}")
        ##        #print(f"  Content: {str(msg.content)[:200]}...")
        ##    # Check for tool calls
        ##    if hasattr(msg, 'tool_calls'):
        ##        print(f"  Tool calls: {msg.tool_calls}")
        ##    if hasattr(msg, 'function_call'):
        ##        print(f"  Function call: {msg.function_call}")
        ##print(f"========================\n")

        for msg in result.messages:
            if isinstance(msg, TextMessage):
                self.messages.append(msg.content)

        if not self.check_response:
            assert isinstance(result.messages[-1], TextMessage)
            return False, result

        answer_invalid = False
        if isinstance(result.messages[-1], TextMessage):
            answer_invalid = self.check_invalid_response(result.messages[-1].content)
        else:
            answer_invalid = True
        retries = 0
        while answer_invalid and retries < self.max_retries:
            new_user_prompt = (
                "The previous response was invalid. Please try again.\n\n" + task
            )
            # print("Retrying with new prompt...")
            result = await agent.run(task=new_user_prompt)
            if isinstance(result.messages[-1], TextMessage):
                answer_invalid = self.check_invalid_response(
                    result.messages[-1].content
                )
            else:
                answer_invalid = True
            retries += 1
        return answer_invalid, result

    async def run(self):
        # TODO: START REMOVE AFTER DEBUG
        # Suppress verbose logging
        import logging
        
        # Reduce DEBUG output from these loggers
        logging.getLogger("openai._base_client").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)
        logging.getLogger("autogen_agentchat.events").setLevel(logging.WARNING)
        
        # Keep autogen_core.events at INFO level so ReasoningCapture still works
        logging.getLogger("autogen_core.events").setLevel(logging.INFO)

        # For autogen_core.events: remove default handlers but keep level at INFO
        # so our ReasoningCapture handler still receives the events
        events_logger = logging.getLogger("autogen_core.events")
        events_logger.setLevel(logging.INFO)
        # Remove all existing handlers from this logger
        events_logger.handlers = []
        # Prevent propagation to root logger (which would print to console)
        events_logger.propagate = False
        # TODO: END REMOVE AFTER DEBUG

        system_prompt = self.experiment_type.get_system_prompt()
        user_prompt = self.experiment_type.get_user_prompt()
        assert (
            user_prompt is not None
        ), "User prompt must be provided for single-turn run."

        assert (
            len(self.servers) > 0
        ), "No MCP servers available. Please provide server_path or server_url."

        wokbenches = [McpWorkbench(server) for server in self.servers]

        # Start the servers
        for workbench in wokbenches:
            await workbench.start()

        # Explicitly collect all tools from workbenches
        all_tools = []
        for wb in wokbenches:
            wb_tools = await wb.list_tools()
            all_tools.extend(wb_tools)
        
        print(f"\n=== DEBUG: Collected {len(all_tools)} tools from MCP ===")
        for tool in all_tools:
            tool_name = tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', 'unknown')
            print(f"  - {tool_name}")
        print(f"===================================\n")

        # Debug: Check what tools are available from MCP
        print(f"\n=== DEBUG: MCP Tools ===")
        for i, wb in enumerate(wokbenches):
            print(f"Workbench {i}:")
            try:
                tools = await wb.list_tools()
                print(f"  Available tools: {len(tools)}")
                for tool in tools:
                    # Tools might be dicts or objects
                    if isinstance(tool, dict):
                        print(f"    - {tool.get('name', 'unknown')}: {tool.get('description', 'No description')[:100]}")
                    else:
                        print(f"    - {tool.name}: {tool.description[:100] if hasattr(tool, 'description') else 'No description'}")
            except Exception as e:
                print(f"  Error listing tools: {e}")
                import traceback
                traceback.print_exc()
        print(f"========================\n")


        # TODO: START DEBUG REASONING
        import logging
        import json
        import io
        import sys
        
        class ReasoningCapture(logging.Handler):
            """Custom log handler to capture and display reasoning in real-time"""
            
            def __init__(self, display_mode="detailed"):
                super().__init__()
                self.reasoning_history = []
                self.display_mode = display_mode
                self.step_counter = 0
            
            def emit(self, record):
                """Called for each log message - displays reasoning immediately"""
                if record.name == "autogen_core.events":
                    try:
                        log_data = json.loads(record.getMessage())
                        
                        if log_data.get("type") == "LLMCall" and "response" in log_data:
                            response = log_data["response"]
                            if "choices" in response:
                                for choice in response["choices"]:
                                    message = choice.get("message", {})
                                    reasoning = message.get("reasoning_content")
                                    tool_calls = message.get("tool_calls", [])
                                    content = message.get("content")
                                    finish_reason = choice.get("finish_reason")
                                    
                                    if reasoning:
                                        self.step_counter += 1
                                        self.reasoning_history.append(reasoning)
                                        self._display_detailed(reasoning, tool_calls, content, finish_reason)
                                        sys.stderr.flush()
                    except (json.JSONDecodeError, KeyError) as e:
                        pass
            
            def _display_detailed(self, reasoning, tool_calls, content, finish_reason):
                """Detailed display with full reasoning"""
                sys.stderr.write("\n" + "="*80 + "\n")
                sys.stderr.write(f"STEP {self.step_counter} - MODEL REASONING\n")
                sys.stderr.write("="*80 + "\n")
                sys.stderr.write(f"{reasoning}\n")
                
                # Always display content if it exists (text response)
                if content:
                    sys.stderr.write("\n" + "-"*80 + "\n")
                    sys.stderr.write("CONTENT OUTPUT:\n")
                    sys.stderr.write("-"*80 + "\n")
                    sys.stderr.write(f"{content}\n")
                
                # Always display tool calls if they exist
                if tool_calls:
                    sys.stderr.write("\n" + "-"*80 + "\n")
                    sys.stderr.write("TOOL CALLS OUTPUT:\n")
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        sys.stderr.write(f"  -> {func.get('name')}({func.get('arguments')})\n")
                
                sys.stderr.write("="*80 + "\n\n")
            
            def get_reasoning_history(self):
                return self.reasoning_history
            
            def clear_reasoning_history(self):
                self.reasoning_history = []
                self.step_counter = 0

        
        # Set up the reasoning capture
        reasoning_capture = ReasoningCapture()
        reasoning_capture.setLevel(logging.INFO)
        logging.getLogger("autogen_core.events").addHandler(reasoning_capture)
        # TODO: END DEBUG REASONING

        # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name="Assistant",
            model_client=self.model_client,
            system_message=system_prompt,
            workbench=wokbenches,
            max_tool_iterations=self.max_tool_calls,
        )

        # TODO: REMOVE START
        # Debug: Check what tools the agent has
        print(f"\n=== DEBUG: Agent Configuration ===")
        print(f"Agent has workbench: {agent._workbench is not None if hasattr(agent, '_workbench') else 'unknown'}")
        print(f"Max tool iterations: {self.max_tool_calls}")
        if hasattr(agent, '_tools'):
            print(f"Number of tools: {len(agent._tools) if agent._tools else 0}")
        print(f"===================================\n")

        from autogen_core import EVENT_LOGGER_NAME
        import logging
        
        # Set up detailed logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        # TODO: REMOVE END


        answer_invalid, result = await self.step(agent, user_prompt)

        ## TODO: REMOVE START
        ## Inspect result structure
        #if hasattr(result, 'messages'):
        #    for msg in result.messages:
        #        print(f"Message: {msg}")
        #        if hasattr(msg, 'content'):
        #            print(f"Content: {msg.content}")
        #        if hasattr(msg, 'metadata'):
        #            print(f"Metadata: {msg.metadata}")

        #all_reasoning = reasoning_capture.get_reasoning_history()
        #print("\n==>> All reasoning steps (including intermediate):")
        #for i, reasoning in enumerate(all_reasoning, 1):
        #    print(f"{i}. {reasoning}")
        ## TODO: REMOVE END

        for workbench in wokbenches:
            await workbench.stop()

        if answer_invalid:
            raise ValueError("Failed to get a valid response after maximum retries.")
        else:
            return result.messages[-1].content

    async def chat(self):
        system_prompt = self.experiment_type.get_system_prompt()

        handoff_termination = HandoffTermination(target="user")
        # Define a termination condition that checks for a specific text mention.
        text_termination = TextMentionTermination("TERMINATE")

        assert (
            len(self.servers) > 0
        ), "No MCP servers available. Please provide server_path or server_url."

        wokbenches = [McpWorkbench(server) for server in self.servers]

        # Start the servers
        for workbench in wokbenches:
            await workbench.start()

        # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name="Assistant",
            model_client=self.model_client,
            system_message=system_prompt,
            workbench=wokbenches,
            max_tool_iterations=self.max_tool_calls,
            reflect_on_tool_use=True,
        )

        user = UserProxyAgent("USER", input_func=input)
        team = RoundRobinGroupChat(
            [agent, user],
            max_turns=self.max_multi_turns,
            # termination_condition=text_termination,
        )

        result = team.run_stream()
        await Console(result)
        for workbench in wokbenches:
            await workbench.stop()

        await self.model_client.close()

    async def refine(self, feedback: str):
        raise NotImplementedError(
            "TODO: Multi-turn refine currently not supported. - S.Z."
        )
