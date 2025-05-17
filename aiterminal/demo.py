import autogen
import os

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Configure model with API key
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": api_key
    }
]

# Define a single agent that will create both files
developer = autogen.AssistantAgent(
    name="developer",
    llm_config={"config_list": config_list},
    system_message="""You are an expert developer who can write both Python and Bash code.
When asked to create files, write the complete code for each file.
Be precise and make sure the code will work correctly."""
)

# Define user proxy agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "use_docker": False,
    }
)

# Run the demo
print("===== Running AutoGen Bash + Python Demo =====")

# Request both Python and Bash files in a single conversation
user_proxy.initiate_chat(
    developer,
    message="""Create two files:

1. A Python file named 'prime_finder.py' that:
   - Implements the Sieve of Eratosthenes algorithm to find primes up to 50
   - Prints the result and execution time

2. A Bash script named 'run_primes.sh' that:
   - Makes the Python file executable (if needed)
   - Runs the Python script
   - Displays a success message

Make sure both files work correctly. Write complete, executable code for both files."""
)

print("\n===== Demo Completed =====")
