import os
import json
import subprocess
import autogen
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API_KEY from the environment
api_key = os.getenv("API_KEY")

# 1. Set LLM Configuration
config_list = [
    {
        'model': 'gpt-4o-mini',
        'api_key': api_key,
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

coder = autogen.AssistantAgent(
    name="Verilog_Coder",
    system_message="You are an expert Verilog coder. Your goal is to:"
    "- Carefully translate user specifications into a precise, synthesizable Verilog module"
    "- Ensure the code exactly matches the provided requirements"
    "- Write clean, well-commented, and efficient code"
    "- Prioritize clarity and correctness over unnecessary complexity",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

critic = autogen.AssistantAgent(
    name="Verilog_Critic",
    system_message="You are a meticulous Verilog code reviewer. Your responsibilities include:"
    "- Thoroughly validate the code against the original specification"
    "- Identify any discrepancies between implementation and requirements"
    "- Provide precise, actionable feedback",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

# 3. Create GroupChat & Manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, critic],
    messages=[],
    max_round=12
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# -------------------------------
# Helper Functions
# -------------------------------
def extract_verilog_code(response_text):
    """Extract Verilog code block from agent response."""
    match = re.search(r"```verilog\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    if "module" in response_text:
        return response_text.strip()
    return "Error: No Verilog code detected"

def chat_with_agent(agent, user_prompt, extract_verilog = True):
    """Interact with an agent and return the extracted Verilog code."""
    groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": user_prompt})
    agent_text = agent.generate_reply(messages=groupchat.messages, user_input=user_prompt)
    if not isinstance(agent_text, str):
        agent_text = str(agent_text)
    groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
    if extract_verilog:
        return extract_verilog_code(agent_text)
    return agent_text

def run_systemverilog_tests(test_filename, ref_filename=None):
    """Run the SystemVerilog test using test and reference files."""
    test_results = []

    def run_command(command):
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

    print("\n=== Running SystemVerilog Simulation ===")
    
    # Compile files command
    compile_cmd = f"iverilog -g2012 -o sim {test_filename} TopModule.v"
    if ref_filename and os.path.exists(ref_filename):
        compile_cmd += f" {ref_filename}"
    
    # Compile files
    output, error = run_command(compile_cmd)
    if error and "error" in error.lower():
        test_results.append(f"❌ Compilation Failed\n{error}")
        return "\n".join(test_results)
    
    # Run simulation
    output, error = run_command("vvp sim")
    if output:
        test_results.append(f"✅ Simulation Output\n{output}")
    if error:
        test_results.append(f"⚠️ Simulation Warning/Error\n{error}")
    
    # Log results
    test_run_number = 1
    if os.path.exists("verilog_test_results.txt"):
        with open("verilog_test_results.txt", "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
            test_run_number = sum(1 for line in existing_lines if "Test Run #" in line) + 1

    with open("verilog_test_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== Test Run #{test_run_number} ===\n")
        f.write("\n".join(test_results) + "\n")

    print("\n✅ All tests completed! Results saved in 'verilog_test_results.txt'.")
    return "\n".join(test_results)

def evaluate_test_results(test_results):
    """
    Evaluate test results.
    """
    results_lower = test_results.lower()
    
    if "mismatches: 0 in " in results_lower:
        return True, "Test passed with no mismatches."
    
    if "has no mismatches" in results_lower:
        return True, "Test passed successfully."
    
    if "error" in results_lower or "fail" in results_lower:
        return False, "Test failed with errors or failures."
    
    if "mismatches:" in results_lower and not "mismatches: 0" in results_lower:
        return False, "Test failed with mismatches."
    
    return False, "Could not determine test result, defaulting to failure."

# -------------------------------
# Main Automated Design Process
# -------------------------------
def orchestrate_design_flow(user_request, test_filename, ref_filename=None):
    """Flexible design process with adaptive implementation."""
    conversation_log = []
    print("\n🚀 Starting Verilog design process...")

    # Direct to Coder: Generate initial implementation
    print("\n🖥 Generating initial Verilog implementation...")
    coder_response = chat_with_agent(
        coder,
        f"Implement a Verilog module based on the following detailed specification:\n\n{user_request}\n\n"
        "Guidelines for implementation:"
        "- Carefully analyze the entire specification"
        "- Create a module that precisely matches all requirements"
        "- Use appropriate Verilog constructs based on the module type"
        "- Write clear, well-commented, and efficient code"
        "- Ensure the implementation is synthesizable and follows best practices"
    )
    conversation_log.append(f"=== Initial Coder Implementation ===\n{coder_response}\n")

    # Test the implementation
    with open("TopModule.v", "w", encoding="utf-8") as f:
        f.write(coder_response)

    test_results = run_systemverilog_tests(test_filename, ref_filename)
    print(f'TEST_RESULTS: {test_results}')
    conversation_log.append(f"=== SystemVerilog Test Results ===\n{test_results}\n")

    # Evaluate test results
    tests_passed, reason = evaluate_test_results(test_results)
    conversation_log.append(f"=== Test Evaluation ===\nResult: {'PASS' if tests_passed else 'FAIL'}\nReason: {reason}\n")

    try:
        with open(test_filename, "r", encoding="utf-8") as f:
            test_file_content = f.read()
    except FileNotFoundError:
        print(f"\n❌ Error: Test file '{test_filename}' not found.")
        return None, conversation_log

    for test_fix_iteration in range(5):
        tests_passed, reason = evaluate_test_results(test_results)
        if tests_passed:
            break

        print(f"\n🔄 Test failed. Refining code based on errors (Iteration {test_fix_iteration+1})...")

        critic_analysis = chat_with_agent(
            critic,
            f"Analyze the test failures based on the following test file:\n{test_filename}\n\n"
            f"Test Failure Details:\n{test_results}\n\n", extract_verilog =  False
            # "Provide insights on:"
            # "- Why the failures occurred based on the test expectations"
            # "- Possible mistakes in the implementation"
            # "- Suggestions on how to correct them", extract_verilog =  False
        )
        conversation_log.append(f"=== Critic Analysis (Iteration {test_fix_iteration+1}) ===\n{critic_analysis}\n")

        coder_response = chat_with_agent(
            coder,
            # f"Refine the Verilog code based on the following feedback from the critic:\n\n"
            f"Original Specification:\n{user_request}\n\n"
            f"Previous Implementation:\n{coder_response}\n\n"
            f"Test results: {test_results}"
            f"Critic Analysis:\n{critic_analysis}\n\n"
            # "Fix Instructions:"
            # "- Address the specific issues found in the critic’s analysis"
            # "- Ensure correctness without breaking other working parts"
            # "- Maintain adherence to the original specification",
            # f"Analyze test results and test cases in {test_file_content} and correct the code.", 
        )
        conversation_log.append(f"=== Refined Code (Test Fix {test_fix_iteration+1}) ===\n{coder_response}\n")

        # Save updated implementation
        with open("TopModule.v", "w", encoding="utf-8") as f:
            f.write(coder_response)

        # Rerun tests
        test_results = run_systemverilog_tests(test_filename, ref_filename)
        conversation_log.append(f"=== SystemVerilog Retest Results (Iteration {test_fix_iteration+1}) ===\n{test_results}\n")

    # Final Evaluation
    tests_passed, reason = evaluate_test_results(test_results)
    conversation_log.append(f"=== Final Test Evaluation ===\nResult: {'PASS' if tests_passed else 'FAIL'}\nReason: {reason}\n")

    if tests_passed:
        print("\n🎉 Verilog module successfully implemented and verified!")
    else:
        print("\n❌ Design process ended with unresolved test failures. Further debugging required.")

    return coder_response, conversation_log



def save_conversation_log(conversation_log):
    with open("design_flow_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_log))

def extract_prompt(prompt_file):
    """
    Reads the Verilog design prompt from a file.
    """
    try:
        with open(prompt_file, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {prompt_file} not found.")
        return None


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Clear previous results
    open("design_flow_results.txt", "w", encoding="utf-8").close()
    open("verilog_test_results.txt", "w", encoding="utf-8").close()
    
    # # Sample prompt for the zero output module
    # user_request = extract_prompt("Prob001_zero_prompt.txt")

    # # Path to the SystemVerilog test file and reference file
    # test_filename = "Prob001_zero_test.sv"
    # ref_filename = "Prob001_zero_ref.sv"

    # user_request_file = "Prob100_fsm3comb_prompt.txt"
    # test_filename = "Prob100_fsm3comb_test.sv"
    # ref_filename = "Prob100_fsm3comb_ref.sv"

    # user_request_file = "Prob156_review2015_fancytimer_prompt.txt"
    # test_filename = "Prob156_review2015_fancytimer_test.sv"
    # ref_filename = "Prob156_review2015_fancytimer_ref.sv"

    user_request_file = "Prob122_kmap4_prompt.txt"
    test_filename = "Prob122_kmap4_test.sv"
    ref_filename = "Prob122_kmap4_ref.sv"

    # Check if the files exist
    files = [user_request_file, test_filename, ref_filename]
    file_names = ["User Request File", "Test file", "Reference file"]

    for file, name in zip(files, file_names):
        if not os.path.exists(file):
            print(f"❌ {name} {file} not found. Please ensure it exists in the current directory.")
            exit(1)

    user_request = extract_prompt(user_request_file)
    
    coder_verilog, conversation_log = orchestrate_design_flow(user_request, test_filename, ref_filename)
    save_conversation_log(conversation_log)

    print("\n✅ Design and testing completed! Full log saved in 'design_flow_results.txt'.")