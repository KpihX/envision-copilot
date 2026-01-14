import sys
print("Starting check...", flush=True)
try:
    from envision_preprocess.builder import NetworkBuilder
    print("Builder imported", flush=True)
    nb = NetworkBuilder()
    print("Builder instantiated", flush=True)
    nb.build()
    print("Build called", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
