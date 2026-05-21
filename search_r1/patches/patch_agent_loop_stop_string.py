"""
Patch tool_agent_loop.py to add </search> as stop string for search_r1 format.
This ensures the model stops generating after <search>query</search>,
allowing tool call extraction before the response length limit is hit.
"""
import sys

filepath = "/root/miniconda3/lib/python3.12/site-packages/verl/experimental/agent_loop/tool_agent_loop.py"

with open(filepath, 'r') as f:
    content = f.read()

# Check if already patched
if "PATCH: add stop strings for search_r1" in content:
    print("Already patched!")
    sys.exit(0)

# Find the generate call in _handle_generating_state and add stop strings before it
old_code = """        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )"""

new_code = """        # PATCH: add stop strings for search_r1 format to prevent model from
        # generating past </search> tag (which would fill response_length and
        # cause termination before tool calls are extracted)
        if self.tool_parser_name == "search_r1":
            sampling_params = dict(sampling_params)
            stop_list = sampling_params.get("stop", []) or []
            if isinstance(stop_list, str):
                stop_list = [stop_list]
            else:
                stop_list = list(stop_list)
            if "</search>" not in stop_list:
                stop_list.append("</search>")
            sampling_params["stop"] = stop_list

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )"""

if old_code not in content:
    print("ERROR: Could not find the target code to patch!")
    print("Looking for generate call pattern...")
    # Try to find it more loosely
    import re
    if "simple_timer" in content and "generate_sequences" in content:
        print("Found simple_timer + generate_sequences, but exact match failed.")
        print("Showing context around it:")
        idx = content.find('simple_timer("generate_sequences"')
        if idx >= 0:
            print(content[max(0,idx-200):idx+400])
    sys.exit(1)

patched = content.replace(old_code, new_code)

with open(filepath, 'w') as f:
    f.write(patched)

print("Successfully patched tool_agent_loop.py!")
print("Added </search> stop string for search_r1 format")
