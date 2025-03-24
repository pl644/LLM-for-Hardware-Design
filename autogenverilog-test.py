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

def chat_with_agent(agent, user_prompt):
    """Interact with an agent and return the extracted Verilog code."""
    groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": user_prompt})
    agent_text = agent.generate_reply(messages=groupchat.messages, user_input=user_prompt)
    if not isinstance(agent_text, str):
        agent_text = str(agent_text)
    groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
    return extract_verilog_code(agent_text)

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

    # Critic Review
    print("\n🔍 Comprehensive Code Review...")
    critic_response = chat_with_agent(
        critic,
        f"Thoroughly review this Verilog implementation against the original specification:\n\n{user_request}\n\n"
        f"Code to review:\n\n{coder_response}\n\n"
        "Review Criteria:"
        "- Verify complete alignment with specification"
        "- Check for any logical inconsistencies"
        "- Provide specific, actionable feedback"
    )
    conversation_log.append(f"=== Critic Detailed Review ===\n{critic_response}\n")

    # Refinement with Maximum 2 Iterations
    for iteration in range(2):
        # Check if further refinement is needed
        if "No issues found" in critic_response or "Looks good" in critic_response:
            break

        print(f"\n🛠 Refining Verilog Code (Iteration {iteration+1})...")
        coder_response = chat_with_agent(
            coder,
            f"Refine the Verilog code based on the following specification and critic feedback:\n\n"
            f"Original Specification:\n{user_request}\n\n"
            f"Previous Implementation:\n{coder_response}\n\n"
            f"Critic Feedback:\n{critic_response}\n\n"
            "Refinement Instructions:"
            "- Address each point in the critic's feedback"
            "- Make improvements if necessary"
        )
        conversation_log.append(f"=== Refined Code (Iteration {iteration+1}) ===\n{coder_response}\n")

        # Re-validate against specification
        critic_response = chat_with_agent(
            critic,
            f"Verify this Verilog implementation exactly matches the original specification:\n\n{user_request}\n\n"
            f"Code to verify:\n\n{coder_response}\n\n"
            "Verification Process:"
            "- Perform a comprehensive review"
            "- Check against each requirement in the original specification"
            "- Identify any remaining discrepancies or potential improvements"
        )
        conversation_log.append(f"=== Verification Review (Iteration {iteration+1}) ===\n{critic_response}\n")

    # Test the implementation
    with open("TopModule.v", "w", encoding="utf-8") as f:
        f.write(coder_response)

    test_results = run_systemverilog_tests(test_filename, ref_filename)
    conversation_log.append(f"=== SystemVerilog Test Results ===\n{test_results}\n")

    # Evaluate test results
    tests_passed, reason = evaluate_test_results(test_results)
    conversation_log.append(f"=== Test Evaluation ===\nResult: {'PASS' if tests_passed else 'FAIL'}\nReason: {reason}\n")

    # Additional refinement if tests fail
    iteration_count = 0
    while not tests_passed and iteration_count < 3:
        iteration_count += 1
        print(f"\n🚨 Testing failed! (Reason: {reason})")
        
        critic_response = chat_with_agent(
            critic,
            f"The Verilog design failed testing. Error details:\n\n{test_results}\n\n"
            f"Test failure reason: {reason}\n\n"
            "Analyze the test results and suggest specific corrections:"
            "- Identify potential implementation issues"
            "- Propose targeted fixes"
            "- Ensure alignment with original specification"
        )
        conversation_log.append(f"=== Critic Post-Test Feedback (Iteration {iteration_count}) ===\n{critic_response}\n")

        coder_response = chat_with_agent(
            coder,
            f"Revise the Verilog implementation based on test failures:\n\n"
            f"Original Specification:\n{user_request}\n\n"
            f"Previous Implementation:\n{coder_response}\n\n"
            f"Critic Feedback:\n{critic_response}\n\n"
            f"Test Results:\n{test_results}\n\n"
            "Directly address the test failures"
        )
        conversation_log.append(f"=== Revised Code (Post-Test {iteration_count}) ===\n{coder_response}\n")

        # Save and test updated design
        with open("TopModule.v", "w", encoding="utf-8") as f:
            f.write(coder_response)

        test_results = run_systemverilog_tests(test_filename, ref_filename)
        conversation_log.append(f"=== SystemVerilog Test Results (Iteration {iteration_count}) ===\n{test_results}\n")
        
        tests_passed, reason = evaluate_test_results(test_results)
        conversation_log.append(f"=== Test Evaluation (Iteration {iteration_count}) ===\nResult: {'PASS' if tests_passed else 'FAIL'}\nReason: {reason}\n")
        
        if tests_passed:
            print(f"🎉 Tests PASSED! (Reason: {reason})")
            break

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

    user_request_file = "Prob100_fsm3comb_prompt.txt"
    test_filename = "Prob100_fsm3comb_test.sv"
    ref_filename = "Prob100_fsm3comb_ref.sv"

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