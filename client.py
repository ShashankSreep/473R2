import asyncio
import sys
from typing import Optional
from ollama import Client
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create a client class to interact with the MCP server
class MCPClient:
    def __init__(self):
        """Initialize the MCP client."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama = Client()
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.
        
        Args:
            server_script_path (str): Path to the server script (.py or .js).
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()

        # Fetch available tools from the server
        response = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                },
            }
            for tool in response.tools
        ]
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.tools])


    # Process a user query using Llama and available tools
    # Invoked when the user types a query in the chat loop => For each query, the client calls process_query
    async def process_query(self, query: str) -> str:
        """Process a user query using Llama and available tools."""
        messages = [{
            "role": "system",
            "content": (
                "You are a helpful assistant. Only use a tool if the user asks for weather or temperature or stocks."
                "For general questions, respond normally without calling a tool."
            )
        },
        {"role": "user", "content": query}]

        # Initial call to Llama
        response = self.ollama.chat(
            model="llama3.1",
            messages=messages,
            tools=self.tools,  # Pass only the available tools
        )

        # If no tool is needed, return the response immediately
        if response.message.content or response.message.tool_calls is None:
            return response.message.content or "I didn't get that."

        tool_results = []
        final_text = []

        # If the response contains tool calls, process them
        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Validate tool existence before calling
                if tool_name not in [t["function"]["name"] for t in self.tools]:
                    final_text.append(f"[Error: Tool '{tool_name}' does not exist!]")
                    continue

                # Execute tool call
                result = await self.session.call_tool(tool_name, dict(tool_args))
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Append tool result as a new user message to continue chat
                messages.append({"role": "user", "content": result.content[0].text})

            # Only call Llama again if a tool was actually used
            if tool_results:
                response = self.ollama.chat(
                    model="llama3.1",
                    messages=messages,
                )
                final_text.append(response.message.content)

        return "\n".join(final_text)


    # Runs the chat loop
    # The loop ends when the user types 'quit'
    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print(response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    """Main function to start the client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
