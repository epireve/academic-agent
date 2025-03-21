# AI Agent Creator Instructions for smolagents Framework

smolagents is a lightweight library from Hugging Face designed to build powerful AI agents with minimal code. It offers simplicity with core logic under 1,000 lines of code while delivering high performance. Your role is to help create efficient AI agents using smolagents' features:

**Official Documentation**: https://github.com/huggingface/smolagents

1. **Planning**: Plan your agent structure, determining which agent type, tools, and model will best solve your requirements.
2. **Installation**: Set up smolagents and required dependencies.
3. **Agent Development**: Create appropriate agents (CodeAgent or ToolCallingAgent).
4. **Model Selection**: Choose and configure the right LLM provider.
5. **Tool Integration**: Add necessary tools from built-in options or custom implementations.
6. **Execution Environment**: Configure secure execution environments if needed.
7. **Deployment**: Share your agent to the Hugging Face Hub or deploy it locally.

# Step 1: Key Features

smolagents stands out with several key features:

- **Simplicity**: Core agent logic fits in ~1,000 lines of code, minimizing abstractions
- **First-class Code Agent support**: Agents write actions directly in Python code
- **Hub integrations**: Easily share and pull tools from the Hugging Face Hub
- **Model-agnostic**: Works with any LLM (local models, Hugging Face models, OpenAI, Anthropic, etc.)
- **Modality-agnostic**: Supports text, vision, video, and audio inputs
- **Tool-agnostic**: Compatible with tools from LangChain, Anthropic's MCP, or Hub Spaces
- **Code execution security**: Options for sandboxed environments via E2B or Docker

# Step 2: Installation

Install smolagents with pip:

```bash
pip install smolagents
```

For development installation:

```bash
git clone https://github.com/huggingface/smolagents.git
cd smolagents
pip install -e ".[dev]"
```

# Step 3: Agent Types

smolagents provides two main agent types:

## CodeAgent

The recommended agent type that writes its actions as Python code snippets. This approach is more efficient and performant than traditional JSON-based tool calling.

```python
from smolagents import CodeAgent

agent = CodeAgent(tools=[...], model=...)
```

## ToolCallingAgent

A traditional agent that uses JSON/text blobs for actions:

```python
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(tools=[...], model=...)
```

# Step 4: Model Selection

smolagents is compatible with multiple LLM providers:

## HfApiModel (Hugging Face Inference API)

Gateway to four inference providers (Hugging Face, Together, Groq, Fireworks):

```python
from smolagents import HfApiModel

model = HfApiModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together",  # Options: "huggingface", "together", "groq", "fireworks"
)
```

## LiteLLMModel (100+ LLMs)

Access to over 100 LLM providers:

```python
from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    temperature=0.2,
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```

## OpenAI-compatible servers

Connect to OpenAI or compatible API servers:

```python
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1/",  # Leave blank for OpenAI servers
    api_key=os.environ["TOGETHER_API_KEY"],
)
```

## Local Transformers models

Run models locally:

```python
from smolagents import TransformersModel

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="auto"
)
```

## Azure OpenAI models

Use Azure-hosted OpenAI models:

```python
import os
from smolagents import AzureOpenAIServerModel

model = AzureOpenAIServerModel(
    model_id=os.environ.get("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("OPENAI_API_VERSION")    
)
```

# Step 5: Tool Integration

You can equip your agent with various tools:

## Built-in Tools

```python
from smolagents import DuckDuckGoSearchTool, WikipediaTool, PythonReplTool

# Add tools when creating the agent
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaTool(),
        PythonReplTool()
    ],
    model=model
)
```

## Custom Tools

Create your own tools:

```python
from smolagents import Tool

class CustomTool(Tool):
    name = "custom_tool"
    description = "Description of what this tool does"
    
    def __call__(self, *args, **kwargs):
        # Implement tool functionality
        return "Tool result"
```

## Hub Tools

Use tools from the Hugging Face Hub:

```python
from smolagents import load_tool_from_hub

weather_tool = load_tool_from_hub("user/weather-tool")
agent = CodeAgent(tools=[weather_tool], model=model)
```

# Step 6: Running Your Agent

The basic pattern for running an agent:

```python
from smolagents import CodeAgent, HfApiModel

# Create model
model = HfApiModel()

# Create agent with tools
agent = CodeAgent(tools=[...], model=model)

# Run the agent with a user query
result = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
print(result)
```

## Secure Code Execution

For CodeAgents, you can configure secure execution environments:

```python
from smolagents import CodeAgent, E2BCodeExecutor

# Use E2B sandboxed environment
agent = CodeAgent(
    tools=[...],
    model=model,
    code_executor=E2BCodeExecutor(api_key="your-e2b-api-key")
)

# Or use Docker
from smolagents import DockerCodeExecutor
agent = CodeAgent(
    tools=[...],
    model=model,
    code_executor=DockerCodeExecutor()
)
```

# Step 7: Command Line Interface

smolagents provides two CLI commands:

## smolagent

A general-purpose CLI for running CodeAgent with various tools:

```bash
smolagent "Plan a trip to Tokyo, Kyoto and Osaka between Mar 28 and Apr 7." \
  --model-type "HfApiModel" \
  --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --imports "pandas numpy" \
  --tools "web_search"
```

## webagent

A specialized web-browsing agent:

```bash
webagent "go to xyz.com/men, get to sale section, click the first clothing item you see. Get the product details, and the price, return them." \
  --model-type "LiteLLMModel" \
  --model-id "gpt-4o"
```

# Step 8: Sharing Your Agent

You can share your agent to the Hugging Face Hub:

```python
agent.push_to_hub("username/my_agent")

# Later, load it:
from smolagents import CodeAgent
agent = CodeAgent.from_hub("username/my_agent")
```

# Best Practices

1. **Choose the Right Agent**:
   - Use CodeAgent for most use cases (more efficient)
   - Use ToolCallingAgent only if you specifically need JSON-based tool calling

2. **Model Selection**:
   - For best performance, benchmarks show DeepSeek-R1 performs well
   - Consider using local models for privacy-sensitive data
   - Balance performance vs. cost when selecting a model

3. **Security**:
   - Always use sandboxed environments (E2B or Docker) for untrusted code execution
   - Use the secure Python interpreter for simple/trusted use cases

4. **Tool Design**:
   - Keep tools focused on single tasks
   - Provide clear documentation for each tool
   - Consider sharing reusable tools on the Hub

5. **Agent Memory**:
   - Be aware that the CodeAgent keeps state in its memory
   - Clear memory when starting new contexts

# How CodeAgent Works

The CodeAgent follows a ReAct pattern but with code actions:

1. User task is added to agent memory
2. Agent generates code action based on memory 
3. Code is executed (tool calls are Python function calls)
4. Execution logs are stored in memory
5. Process repeats until agent calls the `final_answer` tool

This approach has been shown to use 30% fewer steps than traditional agents and achieves higher performance on difficult benchmarks.

# Common Patterns

## Multi-Agent Setup

```python
from smolagents import CodeAgent

researcher = CodeAgent(tools=[...], model=model, name="Researcher")
writer = CodeAgent(tools=[...], model=model, name="Writer")

# Run the researcher first
research_results = researcher.run("Research the impact of AI on healthcare")

# Pass results to the writer
final_report = writer.run(f"Write a report based on this research: {research_results}")
```

## Streaming Responses

```python
from smolagents import CodeAgent

agent = CodeAgent(tools=[...], model=model)

for chunk in agent.run_stream("Your query here"):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from smolagents import CodeAgent

agent = CodeAgent(tools=[...], model=model)

try:
    result = agent.run("Your query")
except Exception as e:
    print(f"Agent error: {e}")
    # Implement fallback behavior
```

# Additional Resources

- Official GitHub Repository: https://github.com/huggingface/smolagents
- Hugging Face Hub (for sharing agents): https://huggingface.co/models
- Benchmarking Code: https://github.com/huggingface/smolagents/tree/main/benchmarks

Remember to always validate your agent's responses and ensure proper security measures when executing code. Agents should be tested thoroughly before deployment in production environments. 