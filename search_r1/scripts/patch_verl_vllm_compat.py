"""
Monkey-patch verl's vllm_async_server.py for vllm 0.8.5 compatibility.
Adds missing methods and wraps calls that use vllm 0.9+ internal APIs.
Run this script on the Blackwell server to patch verl in-place.
"""
import re

TARGET = "/root/miniconda3/lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py"

with open(TARGET, "r") as f:
    content = f.read()

# 1. Already patched: reset_mm_cache (commented out)
# 2. Already patched: get_supported_tasks (hasattr guard)

# 3. Patch wait_for_requests_to_drain method
# The method on self calls self.engine.wait_for_requests_to_drain()
# We need to make it a no-op if the method doesn't exist
old_wait = """    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()"""
new_wait = """    async def wait_for_requests_to_drain(self):
        if hasattr(self.engine, 'wait_for_requests_to_drain'):
            await self.engine.wait_for_requests_to_drain()
        else:
            import asyncio
            await asyncio.sleep(0.1)"""

if old_wait in content:
    content = content.replace(old_wait, new_wait)
    print("Patched: wait_for_requests_to_drain")
else:
    print("SKIP: wait_for_requests_to_drain (already patched or not found)")

# 4. Patch pause_generation and resume_generation
# These are called in update_weights_from_actor context
# Find the block that uses pause_generation and wrap with hasattr
old_pause = """                await self.engine.pause_generation("""
new_pause = """                if hasattr(self.engine, 'pause_generation'):
                    await self.engine.pause_generation("""

# This is trickier because the block has multiple indented lines after.
# Let me use a different approach - add a compat layer at the top of the class

# Actually, let me use a simpler approach: monkeypatch AsyncLLM class
# to add missing methods

compat_code = '''
# === vllm 0.8.5 compatibility patches ===
def _patch_asyncllm_for_085():
    """Add missing methods to AsyncLLM for vllm 0.8.5 compatibility with verl 0.7.1"""
    import asyncio
    from vllm.v1.engine.async_llm import AsyncLLM

    if not hasattr(AsyncLLM, 'wait_for_requests_to_drain'):
        async def _wait_drain(self):
            await asyncio.sleep(0.1)
        AsyncLLM.wait_for_requests_to_drain = _wait_drain

    if not hasattr(AsyncLLM, 'pause_generation'):
        async def _pause(self, **kwargs):
            pass
        AsyncLLM.pause_generation = _pause

    if not hasattr(AsyncLLM, 'resume_generation'):
        async def _resume(self):
            pass
        AsyncLLM.resume_generation = _resume

    if not hasattr(AsyncLLM, 'reset_mm_cache'):
        async def _reset_mm(self):
            pass
        AsyncLLM.reset_mm_cache = _reset_mm

    if not hasattr(AsyncLLM, 'get_supported_tasks'):
        async def _get_tasks(self):
            return ()
        AsyncLLM.get_supported_tasks = _get_tasks

    # Add a fake output_processor if needed
    class _FakeOutputProcessor:
        request_states = {}
        def abort_requests(self, request_ids):
            pass

    _orig_init = AsyncLLM.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, 'output_processor'):
            self.output_processor = _FakeOutputProcessor()
        if not hasattr(self, 'engine_core'):
            class _FakeCore:
                async def abort_requests_async(self, ids):
                    pass
            self.engine_core = _FakeCore()
    AsyncLLM.__init__ = _patched_init

try:
    _patch_asyncllm_for_085()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to patch AsyncLLM for 0.8.5 compat: {e}")
# === end compatibility patches ===
'''

# Insert the compat code right after the imports at the top of the file
# Find a good insertion point
import_end = content.find('\nlogger = ')
if import_end == -1:
    import_end = content.find('\nclass ')

if import_end > 0:
    content = content[:import_end] + '\n' + compat_code + content[import_end:]
    print("Inserted: AsyncLLM compatibility patches")
else:
    print("ERROR: Could not find insertion point")

# 5. Also undo the comment on reset_mm_cache since we now have the monkeypatch
# Actually keep the comment since the monkeypatch handles it

with open(TARGET, "w") as f:
    f.write(content)

print("Done! Patched", TARGET)
