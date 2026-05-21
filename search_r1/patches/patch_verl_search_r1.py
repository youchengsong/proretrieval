"""Patch verl 0.7.1 to add Search-R1 tool parser format.

Search-R1 uses <search>query</search> / <information>results</information> format
instead of hermes <tool_call> format.
"""
import sys

# 1. Patch tool_parser.py to add SearchR1ToolParser
parser_path = "/root/miniconda3/lib/python3.12/site-packages/verl/experimental/agent_loop/tool_parser.py"
with open(parser_path, "r") as f:
    parser_content = f.read()

search_r1_parser = '''

@ToolParser.register("search_r1")
class SearchR1ToolParser(ToolParser):
    """Tool parser for Search-R1 format: <search>query</search>"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        import regex
        self.search_regex = regex.compile(r"<search>(.*?)</search>", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if "<search>" not in text or "</search>" not in text:
            return text, []

        matches = self.search_regex.findall(text)
        function_calls = []
        for match in matches:
            query = match.strip()
            if query:
                function_calls.append(
                    FunctionCall(
                        name="search",
                        arguments=json.dumps({"query_list": [query]}, ensure_ascii=False)
                    )
                )

        content = self.search_regex.sub("", text)
        return content, function_calls
'''

# Add after the last class definition (before EOF or after GptOssToolParser)
if "SearchR1ToolParser" not in parser_content:
    parser_content += search_r1_parser
    with open(parser_path, "w") as f:
        f.write(parser_content)
    print("Added SearchR1ToolParser to tool_parser.py")
else:
    print("SearchR1ToolParser already exists in tool_parser.py")

# 2. Patch tool_agent_loop.py to format search_r1 tool responses as <information> tags
loop_path = "/root/miniconda3/lib/python3.12/site-packages/verl/experimental/agent_loop/tool_agent_loop.py"
with open(loop_path, "r") as f:
    loop_content = f.read()

# Find the gpt-oss special handling and add search_r1 handling after it
old_gpt_oss_block = '''        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:'''

new_gpt_oss_block = '''        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "search_r1":
            # Search-R1 format: wrap tool response in <information> tags
            all_results = []
            for msg in add_messages:
                content = msg.get("content", "")
                if content:
                    all_results.append(content)
            info_text = "\\n".join(all_results)
            tool_response_text = f"<information>{info_text}</information>"
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:'''

if "search_r1" not in loop_content:
    if old_gpt_oss_block in loop_content:
        loop_content = loop_content.replace(old_gpt_oss_block, new_gpt_oss_block)
        with open(loop_path, "w") as f:
            f.write(loop_content)
        print("Added search_r1 response formatting to tool_agent_loop.py")
    else:
        print("WARNING: Could not find gpt-oss block to patch in tool_agent_loop.py")
        print("Trying to find the block...")
        # Try with different whitespace
        import re
        pattern = re.compile(r'if self\.tool_parser_name == "gpt-oss":', re.MULTILINE)
        match = pattern.search(loop_content)
        if match:
            print(f"Found gpt-oss check at position {match.start()}")
            # Print surrounding context
            start = max(0, match.start() - 50)
            end = min(len(loop_content), match.end() + 500)
            print(f"Context:\n{loop_content[start:end]}")
        else:
            print("gpt-oss check not found at all!")
else:
    print("search_r1 handler already exists in tool_agent_loop.py")

print("\nDone patching verl for Search-R1 format!")
