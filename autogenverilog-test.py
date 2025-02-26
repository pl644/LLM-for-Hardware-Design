import os
import json
import subprocess
import autogen
import re

# 1. Set LLM Configuration
config_list = [
    {
        'model': 'gpt-4o-mini',  
        'api_key': 'YOUR APIKEY',
    },
]
os.environ["OAI_CONFIG_LIST"] = json.dumps(config_list)
os.environ["AUTOGEN_USE_DOCKER"] = "0"

llm_config = {"config_list": config_list, "cache_seed": 42}

# 2. Create Agents
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human engineer interacting with the Verilog team.",
    human_input_mode="TERMINATE",
    code_execution_config={"use_docker": False}
)

architect = autogen.AssistantAgent(
    name="Architect",
    system_message="You are an expert in Verilog architecture design. Provide efficient and modular designs."
                   "Explain your design choices clearly with comments when necessary.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

coder = autogen.AssistantAgent(
    name="Coder",
    system_message="You are a Verilog coding expert. Write clean, efficient, and well-commented Verilog code."
                   "Provide explanations where needed, but do not exclude the Verilog implementation.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

critizer = autogen.AssistantAgent(
    name="Critizer",
    system_message="You are responsible for reviewing Verilog code for syntax correctness, logical consistency, "
                   "and efficiency improvements. Provide constructive feedback while preserving the original code.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

tester = autogen.AssistantAgent(
    name="Tester",
    system_message="You are responsible for generating Verilog testbenches to validate functionality."
                   "Provide explanations where necessary to help debugging.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

test_coder = autogen.AssistantAgent(
    name="Test_Coder",
    system_message="You are responsible for improving Verilog testbenches by refining assertions, coverage, "
                   "and self-checking mechanisms. Keep the original structure intact but improve reliability.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

test_critizer = autogen.AssistantAgent(
    name="Test_Critizer",
    system_message="You are responsible for reviewing Verilog testbenches for coverage, correctness, and effectiveness."
                   "Provide feedback on missing cases, logical errors, and automation quality.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

# 3. Create GroupChat & Manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, architect, coder, critizer, tester, test_coder, test_critizer],
    messages=[],
    max_round=12
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# Function to extract Verilog code (if present) while preserving full response
def extract_verilog_code(response_text):
    match = re.search(r"```verilog\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1)

    if "module" in response_text:
        return response_text.strip()

    return "Error: No Verilog code detected"


# Function to interact with an agent
def chat_with_agent(agent, user_prompt):
    """ Interact with a specific agent and log the response. """
    groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": user_prompt})

    agent_text = agent.generate_reply(messages=groupchat.messages, user_input=user_prompt)
    agent_text = str(agent_text) if not isinstance(agent_text, str) else agent_text  # Ensure string format

    groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
    return extract_verilog_code(agent_text)  # Extract Verilog code if present


# Main Automated Design Process
def orchestrate_design_flow(user_request):
    """ Full ALU design process and conversation logging """

    conversation_log = []
    print("\n🚀 Starting Verilog ALU design process...")

    # Step 1: Architect generates ALU initial design
    print("\n📝 Asking Architect for initial ALU design...")
    architect_response = chat_with_agent(
        architect,
        "Provide a Verilog design for a 4-bit ALU, including key functionalities and considerations."
    )
    conversation_log.append(f"=== Architect's Design ===\n{architect_response}\n")
    print("✅ Architect provided an ALU design.\n")

    # Step 2: Tester generates an automated testbench
    print("\n🧪 Generating testbench using Tester...")
    tester_response = chat_with_agent(
        tester,
        f"This is the Architect's design:\n\n{architect_response}\n\n"
        "Generate a corresponding Verilog testbench (using Verilog not SystemVerilog) that automatically verifies all functionalities using assertions and if-else checks."
    )
    conversation_log.append(f"=== Tester’s Initial Testbench ===\n{tester_response}\n")
    print("✅ Tester provided an initial testbench.\n")

    # Step 3: Iterating between Critizer & Coder to refine ALU design 
    coder_response = architect_response  # 初始 Verilog 代码来自 Architect
    for i in range(3):  
        print(f"\n🔍 Critizing ALU design (Iteration {i+1})...")
        critizer_response = chat_with_agent(
            critizer,
            f"Here is the Verilog code written so far:\n\n{coder_response}\n\n"
            "Check for syntax errors, logical issues, and suggest improvements while keeping the original structure."
        )
        conversation_log.append(f"=== Critizer Feedback (Iteration {i+1}) ===\n{critizer_response}\n")
        print(f"✅ Critizer reviewed ALU design (Iteration {i+1}).\n")

        print(f"\n🛠 Refining ALU Verilog code (Iteration {i+1})...")
    
        coder_response = chat_with_agent(
            coder,
            f"Here is the latest Verilog code:\n\n{coder_response}\n\n"
            f"Here is the feedback from the previous review:\n\n{critizer_response}\n\n"
            "Please refine the Verilog code based on the feedback while keeping the original structure intact."
        )
        conversation_log.append(f"=== Coder (Version {i+1}) ===\n{coder_response}\n")
        print(f"✅ Coder updated ALU design (Version {i+1}).\n")

        if "No issues found" in critizer_response or "Looks good" in critizer_response:
            print("🎉 ALU design finalized!\n")
            break  

    # Step 4: Iterating between Test Critizer & Test Coder to refine testbench 
    test_coder_response = tester_response
    for i in range(3):  
        print(f"\n🔍 Critizing testbench (Iteration {i+1})...")
        test_critizer_response = chat_with_agent(
            test_critizer,
            f"Here is the Verilog testbench written so far:\n\n{test_coder_response}\n\n"
            "Check for syntax errors, missing cases, and suggest improvements."
        )
        conversation_log.append(f"=== Test Critizer Feedback (Iteration {i+1}) ===\n{test_critizer_response}\n")
        print(f"✅ Test Critizer reviewed testbench (Iteration {i+1}).\n")

        print(f"\n🛠 Refining testbench (Iteration {i+1})...")
        test_coder_response = chat_with_agent(
            test_coder,
            f"Here is the latest Verilog testbench:\n\n{test_coder_response}\n\n"
            f"Here is the feedback from the previous review:\n\n{test_critizer_response}\n\n"
            "Please refine the testbench based on the feedback while keeping the original structure intact."
        )
        conversation_log.append(f"=== Test Coder (Version {i+1}) ===\n{test_coder_response}\n")
        print(f"✅ Test Coder updated testbench (Version {i+1}).\n")

        if "No issues found" in test_critizer_response or "Looks good" in test_critizer_response:
            print("🎉 Testbench finalized!\n")
            break   

    # Save Verilog & Testbench files
    print("\n💾 Saving Verilog & Testbench files...")
    with open("alu.v", "w", encoding="utf-8") as f:
        f.write(coder_response)

    with open("alu_tb.v", "w", encoding="utf-8") as f:
        f.write(test_coder_response)

    return coder_response, test_coder_response, conversation_log



# Save conversation log to a file
def save_conversation_log(conversation_log):
    with open("design_flow_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_log))



# === Automated Verilog Testing ===
def run_verilog_tests():
    test_results = []

    def run_command(command):
        """ Run shell commands and return output/errors """
        print(f"\n⚙️ Running command: {command}")
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            print(f"🔹 Output:\n{result.stdout}")
            if result.stderr:
                print(f"❌ Error:\n{result.stderr}")
            return result.stdout, result.stderr
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return str(e), None

    print("\n=== 🔍 Verilog Syntax Check ===")
    output, error = run_command("iverilog -tnull alu.v")
    test_results.append("✅ Passed Syntax Check" if not error else f"❌ Failed Syntax Check\n{error}")

    print("\n=== 🏃 Running Verilog Testbench ===")
    output, error = run_command("iverilog -o alu_tb alu.v alu_tb.v")
    if error:
        test_results.append(f"❌ Compilation Failed\n{error}")
    else:
        output, error = run_command("vvp alu_tb")
        test_results.append(f"📝 Simulation Output:\n{output}")

    print("\n=== 🔍 Running Verilator Lint Check ===")
    output, error = run_command("verilator --lint-only alu.v")
    test_results.append("✅ Passed Verilator Lint Check" if not error else f"❌ Failed Verilator Lint Check\n{error}")

    print("\n=== 🔍 Checking Optimization with Yosys ===")
    output, error = run_command("echo 'read_verilog alu.v; synth' | yosys")
    test_results.append("✅ Passed Yosys Optimization Check" if not error else f"❌ Failed Yosys Optimization Check\n{error}")

    with open("design_flow_results.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(test_results))

    print("\n✅ All tests completed! Results saved in 'design_flow_results.txt'.")



# Main Execution
if __name__ == "__main__":
    user_request = "Design a 4-bit ALU in Verilog, write testbenches, and optimize the implementation."

    coder_verilog, tester_testbench, conversation_log = orchestrate_design_flow(user_request)
    save_conversation_log(conversation_log)

    print("\n=== Running Automated Verilog Testing ===")
    run_verilog_tests()
    print("\n✅ Design and testing completed! Full log saved in 'design_flow_results.txt'.")
