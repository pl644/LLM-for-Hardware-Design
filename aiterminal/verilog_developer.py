import autogen
import os
import sys
import time

# Configure API key
config_list = [
    {
        "model": "gpt-4o-mini",
    }
]

# Set up working directory
WORK_DIR = "verilog_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

# Define Verilog developer agent
verilog_developer = autogen.AssistantAgent(
    name="verilog_developer",
    llm_config={"config_list": config_list},
    system_message="""You are a Verilog developer. When asked to create Verilog designs:
1. Create a single bash script that:
   - MUST START with #!/bin/bash as the first line
   - Creates the Verilog module files (saving them with proper .v extension)
   - Creates a testbench file for simulation
   - Runs iverilog to compile the code
   - Uses vvp to run the simulation
   - Captures and displays the output
   - SAVES the simulation output to a file named "simulation_output.txt"
2. DO NOT provide separate code blocks for Verilog files - embed everything in the bash script
3. Use clear status messages in the bash script to indicate progress
4. Add proper error handling to catch and report issues with any step
5. Respond with ONLY the bash script in a single code block, with no explanations outside the code block
6. Make sure the script is CLEARLY identified as bash by starting with #!/bin/bash"""
)

# Define user proxy for simulation (without custom streaming)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": WORK_DIR,
        "use_docker": False,
        "last_n_messages": 1
    }
)

# Run the simulation
def run_simulation():
    print("===== 正在執行 VERILOG 開發者代理 =====")
    
    # Initialize the chat with a specific request for simulation
    user_proxy.initiate_chat(
        verilog_developer,
        message="""Create a Verilog implementation of a simple 4-bit counter with the following specifications:
- 4-bit output that counts up on each clock cycle
- Reset input to clear the counter
- Enable input to control whether counting occurs

Please create a bash script that will:
1. Generate the Verilog files (counter module and testbench)
2. Compile them with iverilog
3. Run the simulation with vvp
4. Display the simulation results
5. Save the output to 'simulation_output.txt' for later analysis
6. Do NOT delete the simulation_output.txt file when cleaning up

IMPORTANT:
- Send ONLY a bash script in your response
- Make sure the script starts with #!/bin/bash"""
    )
    
    # Print the chat history
    for msg in user_proxy.chat_messages[verilog_developer.name]:
        print(f"\n{msg['role'].upper()}:")
        print(f"{msg['content']}")
        print("-" * 50)
    
    # Check if simulation_output.txt exists
    output_file = os.path.join(WORK_DIR, "simulation_output.txt")
    if os.path.exists(output_file):
        print(f"\n✅ 模擬已完成，結果已保存到 {output_file}")
        return True
    else:
        print(f"\n❌ 找不到模擬輸出文件: {output_file}")
        print("檢查執行過程中是否有錯誤。")
        return False

if __name__ == "__main__":
    success = run_simulation()
    # Return exit code based on success
    sys.exit(0 if success else 1)