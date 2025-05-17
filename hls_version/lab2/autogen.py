from autogen import AssistantAgent, UserProxyAgent
import os
import signal
import sys
import threading
import time

# Define the configuration
config_list = [
    {
        "model": "gpt-4o-mini", 
    }
]

# Recovery function: Restore files from base folder
def recover_from_base():
    print("Recovering files from base folder...")
    base_dir = "base"
    recover_map = {
        "digitrec_base.cpp": "digitrec.cpp",
        "digitrec_base.h": "digitrec.h",
        "digitrec_test_base.cpp": "digitrec_test.cpp",
        "Makefile_base": "Makefile",
        "README_base.md": "README.md",
        "run_base.tcl": "run.tcl",
        "training_data_base.h": "training_data.h",
        "typedefs_base.h": "typedefs.h"
    }

    for src, dst in recover_map.items():
        src_path = os.path.join(base_dir, src)
        if os.path.exists(src_path):
            with open(src_path, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
    
    # Recover script folder
    script_base_path = os.path.join(base_dir, "script_base")
    if os.path.exists(script_base_path):
        # Remove old script folder if exists
        if os.path.exists("script"):
            os.system("rm -rf script")
        os.system(f"cp -r {script_base_path} script")
    print("Recovery complete.")

# Function to check if task is completed
def is_task_completed():
    if os.path.exists("status.txt"):
        with open("status.txt", "r") as f:
            content = f.read().strip()
            if content == "TASK_COMPLETED":
                return True
    return False

# Create the expert agent
expert_agent = AssistantAgent(
    name="Project_Expert",
    system_message="""You are an expert in software development and problem-solving.
    
    First, explore the environment to understand what files are available.
    Then, develop and execute a strategy based on what you find, without making assumptions about build systems or tools.
    
    Your workflow should be:
    1. Explore the project structure using bash commands like 'ls', 'find', 'cat', etc.
    2. Analyze requirements by examining README files and source code
    3. Develop an execution strategy without assuming specific tools exist, mainly focus on complete digitrec.cpp 
    4. Guide implementation step by step, adapting based on feedback
    5. Never modify the existing test cases or files unless explicitly instructed
    6. Makefile and tcl maybe not the correct code, modify them if necessary

    When implementing solutions, use bash commands:
    - For small changes: use echo commands with >> to append or > to overwrite
    - For larger files: use cat << 'EOF' > filename ... EOF

    When you believe the task is complete, write:
    ```bash
    echo "TASK_COMPLETED" > status.txt
    ```
    """,
    llm_config={"config_list": config_list}
)

# Create the environment agent
environment_agent = UserProxyAgent(
    name="Environment_Agent",
    human_input_mode="NEVER",
    system_message="""You are an environment agent that executes commands and reports results.
    MUST execute the bash commands (do not use c++ or makefile) from the expert agent and report the output.
    Do not make suggestions or provide your own analysis - your role is only to execute commands and return results. 
    Never modify the existing test cases or files unless explicitly instructed
    """,
    code_execution_config={"work_dir": ".", "use_docker": False}
)

# Main workflow function
def main():
    # Restore from base before doing anything
    recover_from_base()

    # Remove any existing status file
    if os.path.exists("status.txt"):
        os.remove("status.txt")

    # Function to monitor completion
    def monitor_completion():
        while True:
            if is_task_completed():
                print("\nTask completed! Terminating...")
                os.kill(os.getpid(), signal.SIGTERM)
                break
            time.sleep(5)

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_completion)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Start the conversation with a simple prompt
    initial_message = """Let's begin analyzing and working on this project.
    
    First, explore the environment to understand what's available, then develop an appropriate strategy.
    Start by listing the directory contents to see what we're working with.
    """

    # Let the agents handle the rest
    environment_agent.initiate_chat(
        expert_agent,
        message=initial_message
    )

if __name__ == "__main__":
    try:
        main()
        # Wait for completion
        print("Workflow initiated. Waiting for completion...")
        while not is_task_completed():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Checking completion status...")
        if is_task_completed():
            print("Task was completed successfully.")
        else:
            print("Task was not completed.")
    except Exception as e:
        print(f"Error: {e}")
