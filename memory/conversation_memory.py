from langgraph.checkpoint.memory import MemorySaver

# Single shared checkpointer instance for the app lifetime
checkpointer = MemorySaver()
