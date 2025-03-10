import os
import json
import subprocess
import autogen
import re

# 1. Set LLM Configuration
config_list = [
    {
        'model': 'gpt-4o-mini',
        'api_key': 'YOURAPIKEY',
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
    system_message="You are an expert in Verilog architecture design. Provide efficient and modular designs. "
                   "Explain your design choices clearly with comments when necessary.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

coder = autogen.AssistantAgent(
    name="Coder",
    system_message="You are a Verilog coding expert. Write clean, efficient, and well-commented Verilog code. "
                   "Provide explanations where needed, but do not exclude the Verilog implementation.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="You are responsible for reviewing Verilog code for syntax correctness, logical consistency, "
                   "and efficiency improvements. Provide constructive feedback while preserving the original code.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

tester = autogen.AssistantAgent(
    name="Tester",
    system_message="You are responsible for generating Verilog testbenches to validate functionality. "
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

test_critic = autogen.AssistantAgent(
    name="Test_Critic",
    system_message="You are responsible for reviewing Verilog testbenches for coverage, correctness, and effectiveness. "
                   "Provide feedback on missing cases, logical errors, and automation quality.",
    llm_config=llm_config,
    code_execution_config={"use_docker": False}
)

# 3. Create GroupChat & Manager
groupchat = autogen.GroupChat(
    agents=[user_proxy, architect, coder, critic, tester, test_coder, test_critic],
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

def compute_error_metric(test_results):
    """
    Compute a simple error count based on error indicators.
    This metric is lower if there are fewer errors.
    """
    error_count = 0
    for line in test_results.splitlines():
        if "❌" in line or "error:" in line.lower():
            error_count += 1
    return error_count

def evaluate_version(test_results):
    """
    Multi-dimensional evaluation: combine error count and other quality metrics.
    We use a composite score where higher is better.
    - Base score: 10 minus error_count (error penalty).
    - Bonus for simulation output, lint, and yosys checks.
    """
    error_count = compute_error_metric(test_results)
    simulation_bonus = 5 if "Simulation Output:" in test_results else 0
    lint_bonus = 3 if "✅ Passed Verilator Lint Check" in test_results else 0
    yosys_bonus = 3 if "✅ Passed Yosys Optimization Check" in test_results else 0
    composite_score = (10 - error_count) + simulation_bonus + lint_bonus + yosys_bonus
    return composite_score

def chat_with_agent(agent, user_prompt):
    """Interact with an agent and return the extracted Verilog code."""
    groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": user_prompt})
    agent_text = agent.generate_reply(messages=groupchat.messages, user_input=user_prompt)
    if not isinstance(agent_text, str):
        agent_text = str(agent_text)
    groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
    return extract_verilog_code(agent_text)

def run_verilog_tests():
    """Run Verilog tests and return the concatenated test results."""
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

    print("\n=== Running Automated Verilog Testing ===")

    print("\n=== 🔍 Verilog Syntax Check ===")
    output, error = run_command("iverilog -tnull design.v")
    test_results.append("✅ Passed Syntax Check" if not error else f"❌ Failed Syntax Check\n{error}")

    print("\n=== 🏃 Running Verilog Testbench ===")
    output, error = run_command("iverilog -o design_tb design.v design_tb.v")
    if error:
        test_results.append(f"❌ Compilation Failed\n{error}")
    else:
        output, error = run_command("vvp design_tb")
        test_results.append(f"📝 Simulation Output:\n{output}")

    print("\n=== 🔍 Running Verilator Lint Check ===")
    output, error = run_command("verilator --lint-only design.v")
    test_results.append("✅ Passed Verilator Lint Check" if not error else f"❌ Failed Verilator Lint Check\n{error}")

    print("\n=== 🔍 Checking Optimization with Yosys ===")
    output, error = run_command("echo 'read_verilog design.v; synth' | yosys")
    test_results.append("✅ Passed Yosys Optimization Check" if not error else f"❌ Failed Yosys Optimization Check\n{error}")

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
    Rule-based evaluation of test results.
    Returns a tuple (pass:bool, reason:str, confidence:int).
    """
    results_lower = test_results.lower()
    fatal_errors = ["compilation failed", "segmentation fault"]
    for error in fatal_errors:
        if error in results_lower:
            return False, f"Fatal error detected: '{error}'", 95

    fail_count = results_lower.count("❌ failed")
    passed_tests_pattern = re.search(r"passed:\s*(\d+)", results_lower)
    passed_tests = int(passed_tests_pattern.group(1)) if passed_tests_pattern else None

    if fail_count > 0:
        return False, f"Detected {fail_count} failure indicators", 80
    if passed_tests is not None and fail_count == 0:
        return True, f"Summary shows {passed_tests} tests passed and 0 failed", 90
    return False, "Could not confidently determine test results, defaulting to failure", 30

def evaluate_test_results_with_ai(test_results, llm_config):
    """
    Use AI to evaluate test results.
    This function now checks if a valid response is received.
    """
    evaluator = autogen.AssistantAgent(
        name="Test_Evaluator",
        system_message=(
            "You are a Verilog test results evaluation expert. Analyze test output and determine if the tests passed successfully. "
            "Provide your judgment ('PASS' or 'FAIL') with detailed reasoning."
        ),
        llm_config=llm_config
    )
    ai_user = autogen.UserProxyAgent(
        name="User_proxy",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    message = (
        "Below are the Verilog test output results. Please evaluate these results and determine if the tests passed. "
        "Provide your judgment ('PASS' or 'FAIL') and explain your reasoning.\n\n"
        f"Test Results:\n```\n{test_results}\n```\n\n"
        "Start your response with 'TEST_RESULT: PASS' or 'TEST_RESULT: FAIL', then explain your reasoning."
    )
    ai_user.initiate_chat(evaluator, message=message)
    
    # Check if any response is received
    if evaluator.name not in ai_user.chat_messages or not ai_user.chat_messages[evaluator.name]:
        return False, "No response received from Test_Evaluator, defaulting to failure."
    
    response = ai_user.chat_messages[evaluator.name][-1]["content"]
    if not response.strip():
        return False, "Empty response from Test_Evaluator, defaulting to failure."
    
    result_match = re.search(r"TEST_RESULT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if result_match:
        result = (result_match.group(1).upper() == "PASS")
        explanation = response.split(result_match.group(0), 1)[1].strip()
        return result, explanation
    else:
        if "pass" in response.lower() and "fail" not in response.lower():
            return True, "AI evaluation indicates tests passed, but no explicit marker. Full response: " + response
        else:
            return False, "AI evaluation did not provide clear result, judging as failure. Full response: " + response

def hybrid_test_evaluation(test_results, llm_config, confidence_threshold=70):
    """
    Hybrid evaluation that uses rule-based evaluation first and falls back to AI if necessary.
    """
    result, reason, confidence = evaluate_test_results(test_results)
    if confidence >= confidence_threshold:
        return result, reason, "rules"
    ai_result, ai_reason = evaluate_test_results_with_ai(test_results, llm_config)
    return ai_result, ai_reason, "ai"

# -------------------------------
# Main Automated Design Process
# -------------------------------
def orchestrate_design_flow(user_request):
    """Full design process based on the user request."""
    conversation_log = []
    print("\n🚀 Starting Verilog design process...")

    # Step 1: Architect generates initial design
    print("\n📝 Asking Architect for initial design...")
    architect_response = chat_with_agent(
        architect,
        f"Please provide a Verilog design based on the following user request:\n\n{user_request}\n\n"
        "Include key functionalities, design considerations, and modular implementation where applicable."
    )
    conversation_log.append(f"=== Architect's Design ===\n{architect_response}\n")
    print("✅ Architect provided an initial design.\n")

    # Step 2: Tester generates an automated testbench
    print("\n🧪 Generating testbench using Tester...")
    tester_response = chat_with_agent(
        tester,
        f"This is the Architect's design:\n\n{architect_response}\n\n"
        "Generate a corresponding Verilog testbench (using Verilog, not SystemVerilog) that automatically verifies all functionalities using assertions and if-else checks.\n\n"
        "Ensure that at the end of the testbench, the output explicitly states whether the tests have PASSED or FAILED.\n"
    )
    conversation_log.append(f"=== Tester's Initial Testbench ===\n{tester_response}\n")
    print("✅ Tester provided an initial testbench.\n")

    # Step 3: Iterative refinement for design between Critic & Coder
    coder_response = architect_response
    for i in range(3):
        print(f"\n🔍 Critic reviewing design (Iteration {i+1})...")
        critic_response = chat_with_agent(
            critic,
            f"Here is the Verilog code written so far:\n\n{coder_response}\n\n"
            "Check for syntax errors, logical issues, and suggest improvements while keeping the original structure."
        )
        conversation_log.append(f"=== Critic Feedback (Iteration {i+1}) ===\n{critic_response}\n")
        print(f"✅ Critic reviewed design (Iteration {i+1}).\n")

        print(f"\n🛠 Refining Verilog code (Iteration {i+1})...")
        coder_response = chat_with_agent(
            coder,
            f"Here is the latest Verilog code:\n\n{coder_response}\n\n"
            f"Here is the feedback from the previous review:\n\n{critic_response}\n\n"
            "Please refine the Verilog code based on the feedback while keeping the original structure intact."
        )
        conversation_log.append(f"=== Coder (Version {i+1}) ===\n{coder_response}\n")
        print(f"✅ Coder updated design (Version {i+1}).\n")

        if "No issues found" in critic_response or "Looks good" in critic_response:
            print("🎉 Design finalized!\n")
            break

    # Step 4: Iterative refinement for testbench between Test Critic & Test Coder
    test_coder_response = tester_response
    for i in range(3):
        print(f"\n🔍 Critic reviewing testbench (Iteration {i+1})...")
        test_critic_response = chat_with_agent(
            test_critic,
            f"Here is the Verilog testbench written so far:\n\n{test_coder_response}\n\n"
            "Check for syntax errors, missing cases, and suggest improvements."
        )
        conversation_log.append(f"=== Test Critic Feedback (Iteration {i+1}) ===\n{test_critic_response}\n")
        print(f"✅ Test Critic reviewed testbench (Iteration {i+1}).\n")

        print(f"\n🛠 Refining testbench (Iteration {i+1})...")
        test_coder_response = chat_with_agent(
            test_coder,
            f"Here is the latest Verilog testbench:\n\n{test_coder_response}\n\n"
            f"Here is the feedback from the previous review:\n\n{test_critic_response}\n\n"
            "Please refine the testbench based on the feedback while keeping the original structure intact.\n\n"
            "Ensure that at the end of the testbench, the output explicitly states whether the tests have PASSED or FAILED.\n"
        )
        conversation_log.append(f"=== Test Coder (Version {i+1}) ===\n{test_coder_response}\n")
        print(f"✅ Test Coder updated testbench (Version {i+1}).\n")

        if "No issues found" in test_critic_response or "Looks good" in test_critic_response:
            print("🎉 Testbench finalized!\n")
            break

    # Save the current design and testbench files
    print("\n💾 Saving Verilog & Testbench files...")
    with open("design.v", "w", encoding="utf-8") as f:
        f.write(coder_response)
    with open("design_tb.v", "w", encoding="utf-8") as f:
        f.write(test_coder_response)

    # Run tests on the initial version and set baseline metrics and version history
    test_results = run_verilog_tests()
    best_composite_score = evaluate_version(test_results)
    best_design_code = coder_response
    best_testbench_code = test_coder_response
    version_history = []
    version_history.append({
        "iteration": 0,
        "design": coder_response,
        "testbench": test_coder_response,
        "composite_score": best_composite_score,
        "test_results": test_results
    })
    tolerance_threshold = 2  # Initial tolerance threshold for composite score improvement

    print(f"\nInitial composite score: {best_composite_score}")

    # Step 5: Hybrid Test Evaluation & Iterative Refinement with Adaptive Tolerance
    for i in range(5):
        tests_passed, reason, method = hybrid_test_evaluation(test_results, llm_config)
        if tests_passed:
            print(f"🎉 Tests PASSED! ({method} evaluation: {reason})")
            print("🎉 Design and Testbench finalized!\n")
            conversation_log.append(f"=== Test Evaluation ===\nMethod: {method}\nResult: PASS\nReason: {reason}\n")
            break

        print(f"\n🚨 Testing failed! ({method} evaluation: {reason})")
        print("Sending error logs to Critic and Test_Critic for further refinement...")

        critic_response = chat_with_agent(
            critic,
            f"The Verilog design failed testing. Here is the error output:\n\n{test_results}\n\n"
            f"Test evaluation ({method}) determined failure for reason: {reason}\n\n"
            "Analyze the error and suggest corrections to the design."
        )
        conversation_log.append(f"=== Critic Post-Test Feedback (Iteration {i+1}) ===\n{critic_response}\n")
        print("✅ Critic analyzed testing errors.\n")

        test_critic_response = chat_with_agent(
            test_critic,
            f"The Verilog testbench failed testing. Here is the error output:\n\n{test_results}\n\n"
            f"Test evaluation ({method}) determined failure for reason: {reason}\n\n"
            "Analyze the error and suggest corrections to the testbench."
        )
        conversation_log.append(f"=== Test Critic Post-Test Feedback (Iteration {i+1}) ===\n{test_critic_response}\n")
        print("✅ Test Critic analyzed testbench errors.\n")

        print("\n🛠 Refining design...")
        coder_response = chat_with_agent(
            coder,
            f"Here is the latest Verilog code:\n\n{coder_response}\n\n"
            f"Here is the feedback from Critic:\n\n{critic_response}\n\n"
            "Please refine the Verilog code based on the feedback."
        )
        conversation_log.append(f"=== Coder (Post-Test Version {i+1}) ===\n{coder_response}\n")
        print(f"✅ Coder updated design (Version {i+1}).\n")

        print("\n🛠 Refining testbench...")
        test_coder_response = chat_with_agent(
            test_coder,
            f"Here is the latest testbench:\n\n{test_coder_response}\n\n"
            f"Here is the feedback from Test Critic:\n\n{test_critic_response}\n\n"
            "Please refine the testbench based on the feedback."
        )
        conversation_log.append(f"=== Test Coder (Post-Test Version {i+1}) ===\n{test_coder_response}\n")
        print(f"✅ Test Coder updated testbench (Version {i+1}).\n")

        # Save updated files
        with open("design.v", "w", encoding="utf-8") as f:
            f.write(coder_response)
        with open("design_tb.v", "w", encoding="utf-8") as f:
            f.write(test_coder_response)

        print("\n=== Re-running Verilog Tests ===")
        test_results = run_verilog_tests()
        current_composite_score = evaluate_version(test_results)
        print(f"Current composite score: {current_composite_score}")

        # Adaptive mechanism: adjust tolerance threshold based on improvement or not.
        if current_composite_score > best_composite_score + tolerance_threshold:
            best_composite_score = current_composite_score
            best_design_code = coder_response
            best_testbench_code = test_coder_response
            version_history.append({
                "iteration": i+1,
                "design": coder_response,
                "testbench": test_coder_response,
                "composite_score": best_composite_score,
                "test_results": test_results
            })
            print("Improvement detected. Updating best stable version.")
            tolerance_threshold = max(1, tolerance_threshold - 0.5)  # tighten threshold on improvement
        else:
            print("\nNo significant improvement detected. Increasing tolerance threshold and reverting to previous stable version.")
            tolerance_threshold = min(4, tolerance_threshold + 0.5)  # relax threshold if needed
            coder_response = best_design_code
            test_coder_response = best_testbench_code
            break

    # Optionally, save version history details
    with open("version_history.txt", "w", encoding="utf-8") as f:
        for entry in version_history:
            f.write(f"Iteration {entry['iteration']}, Composite Score: {entry['composite_score']}\n")
            f.write("---- Test Results ----\n")
            f.write(entry["test_results"] + "\n")
            f.write("----------------------\n\n")

    return coder_response, test_coder_response, conversation_log

def save_conversation_log(conversation_log):
    with open("design_flow_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_log))

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    open("design_flow_results.txt", "w", encoding="utf-8").close()
    open("verilog_test_results.txt", "w", encoding="utf-8").close()
    open("design.v", "w", encoding="utf-8").close()
    open("design_tb.v", "w", encoding="utf-8").close()

    user_request = "Design a 4-bit counter in Verilog, write testbenches, and optimize the implementation."
    coder_verilog, tester_testbench, conversation_log = orchestrate_design_flow(user_request)
    save_conversation_log(conversation_log)

    print("\n✅ Design and testing completed! Full log saved in 'design_flow_results.txt'.")
