import os
import json
import subprocess
import autogen
import re
import dotenv
from pathlib import Path
import shutil
import time
from dataclasses import dataclass


# Load API key from environment variable or .env file
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define design metrics for evaluation
@dataclass
class DesignMetrics:
    """Track metrics for Verilog design quality"""
    syntax_errors: int = 0
    simulation_status: bool = False
    latch_count: int = 0
    flip_flop_count: int = 0
    lut_count: int = 0
    critical_path_ns: float = 0
    coverage_percent: float = 0
    error_count: int = 0
    warning_count: int = 0
    optimization_level: int = 0
    
    def __str__(self):
        return (
            f"Syntax Errors: {self.syntax_errors}\n"
            f"Simulation: {'Passed' if self.simulation_status else 'Failed'}\n"
            f"Resources: {self.flip_flop_count} FFs, {self.lut_count} LUTs\n"
            f"Timing: {self.critical_path_ns:.2f} ns critical path\n"
            f"Coverage: {self.coverage_percent:.1f}%\n"
            f"Errors: {self.error_count}, Warnings: {self.warning_count}\n"
            f"Optimization Level: {self.optimization_level}/10"
        )


# 1. Set LLM Configuration - More secure way to handle API keys
def setup_llm_config(model_name="gpt-4o-mini"):
    """Setup LLM configuration with proper API key handling"""
    if not OPENAI_API_KEY:
        raise ValueError("API key not found. Set OPENAI_API_KEY environment variable or add it to .env file")
    
    config_list = [
        {
            'model': model_name,
            'api_key': OPENAI_API_KEY,
        },
    ]
    os.environ["OAI_CONFIG_LIST"] = json.dumps(config_list)
    os.environ["AUTOGEN_USE_DOCKER"] = "0"
    
    return {"config_list": config_list, "cache_seed": 42}


# 2. Create Agents - Enhanced with specialized roles
def create_agents(llm_config):
    """Create and return all required agents with specialized capabilities"""
    # Common configuration for all assistant agents
    assistant_config = {
        "llm_config": llm_config,
        "code_execution_config": {"use_docker": False}
    }
    
    agents = {
        "user_proxy": autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human engineer interacting with the Verilog team.",
            human_input_mode="TERMINATE",
            code_execution_config={"use_docker": False}
        ),
        
        "architect": autogen.AssistantAgent(
            name="Architect",
            system_message="""You are an expert in Verilog architecture design with decades of experience.
            Provide efficient, scalable, and modular designs focused on reusability.
            Apply design patterns like parameterization, clock domain crossing techniques, and pipeline architectures when appropriate.
            Use clear naming conventions, organized hierarchy, and explain your architectural decisions.""",
            **assistant_config
        ),
        
        "coder": autogen.AssistantAgent(
            name="Coder",
            system_message="""You are a Verilog coding expert specialized in writing clean, efficient, and synthesis-friendly code.
            Always use non-blocking assignments (<=) for sequential logic and blocking (=) for combinational.
            Include detailed port descriptions, ensure complete sensitivity lists, and avoid latches.
            Follow best practices for synchronous design, reset handling, and FSM implementations.
            Provide detailed comments explaining your implementation decisions.""",
            **assistant_config
        ),
        
        "critic": autogen.AssistantAgent(
            name="Critic",
            system_message="""You are a Verilog verification expert focused on design quality.
            Check for common issues: incomplete sensitivity lists, missing defaults, race conditions, latches, and timing violations.
            Verify signal widths match, ensure proper parameter usage, and check reset logic.
            Focus on synthesizability, timing closure risks, and potential simulation/synthesis mismatches.
            Provide specific line-by-line feedback with clear remediation steps.""",
            **assistant_config
        ),
        
        "tester": autogen.AssistantAgent(
            name="Tester",
            system_message="""You are an expert in Verilog testbench development.
            Create comprehensive self-checking testbenches with automated stimulus generation and result verification.
            Use tasks for test organization, assertions for verification, and automatic checking for pass/fail status.
            Include corner cases, boundary conditions, and randomized inputs when appropriate.
            Provide clear debug messages and timing diagrams where helpful.""",
            **assistant_config
        ),
        
        "optimizer": autogen.AssistantAgent(
            name="Optimizer",
            system_message="""You are a Verilog optimization specialist with synthesis expertise.
            Identify and resolve timing-critical paths, reduce area usage, and optimize power consumption.
            Improve resource sharing, eliminate redundant logic, and refine FSM encodings.
            Apply advanced pipelining, retiming, and scheduling techniques to meet performance goals.
            Recognize synthesis tool behaviors and optimize code for better inference.""",
            **assistant_config
        ),
        
        "formal_verifier": autogen.AssistantAgent(
            name="Formal_Verifier",
            system_message="""You are a formal verification expert for Verilog designs.
            Define properties, assertions, and invariants to mathematically prove design correctness.
            Identify potential deadlocks, livelocks, and unreachable states in state machines.
            Create constraints that accurately capture the intended operating conditions.
            Focus on exhaustive verification of critical properties and interface contracts.""",
            **assistant_config
        )
    }
    
    return agents


# 3. Setup chat environment
def setup_chat_environment(agents):
    """Setup group chat and manager"""
    agent_list = list(agents.values())
    groupchat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=15  # Increased for more thorough design exploration
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=agents["architect"].llm_config)
    
    return groupchat, manager


# Helper Functions - Enhanced for better analysis
def extract_verilog_code(response_text):
    """Extract Verilog code block from agent response."""
    match = re.search(r"```verilog\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    if "module" in response_text:
        return response_text.strip()
    return None


def extract_metrics_from_output(output_text):
    """Extract design metrics from tool outputs with improved failure detection"""
    metrics = DesignMetrics()
    
    # Extract syntax errors
    metrics.syntax_errors = output_text.lower().count("error:")
    
    # Extract warnings
    metrics.warning_count = output_text.lower().count("warning:")
    
    # Enhanced failure detection
    has_fail_messages = re.search(r"FAIL:|failed:|test\s+failed", output_text, re.IGNORECASE) is not None
    has_pass_messages = "all tests passed" in output_text.lower() or "test passed" in output_text.lower()
    
    # Only consider simulation passed if no failures are detected and we have passing messages
    metrics.simulation_status = has_pass_messages and not has_fail_messages
    
    # Set error count explicitly if test failures found
    if has_fail_messages:
        metrics.error_count += output_text.lower().count("fail")
    
    # Original extraction logic continues...
    ff_match = re.search(r"(\d+)\s+(?:DFF|Flip-flops)", output_text, re.IGNORECASE)
    if ff_match:
        metrics.flip_flop_count = int(ff_match.group(1))
    
    lut_match = re.search(r"(\d+)\s+(?:LUT|LCs|Logic cells)", output_text, re.IGNORECASE)
    if lut_match:
        metrics.lut_count = int(lut_match.group(1))
    
    coverage_match = re.search(r"coverage:\s*(\d+\.?\d*)%", output_text, re.IGNORECASE)
    if coverage_match:
        metrics.coverage_percent = float(coverage_match.group(1))
    
    metrics.error_count += output_text.lower().count("error:")
    metrics.optimization_level = min(10, max(0, 10 - metrics.warning_count//2 - metrics.error_count*2))
    
    return metrics


def run_command(command, description="", timeout=60):
    """Run shell command and return results with proper error handling"""
    print(f"\n⚙️ Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)
        if result.stdout:
            print(f"🔹 Output:\n{result.stdout[:200]}..." if len(result.stdout) > 200 else f"🔹 Output:\n{result.stdout}")
        if result.stderr:
            print(f"❌ Error:\n{result.stderr[:200]}..." if len(result.stderr) > 200 else f"❌ Error:\n{result.stderr}")
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 60 seconds")
        return "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return "", str(e)


def run_verilog_design_tools(output_dir="./outputs"):
    """Run comprehensive suite of Verilog design tools and return metrics"""
    design_file = f"{output_dir}/design.v"
    testbench_file = f"{output_dir}/design_tb.v"
    test_results = []
    outputs = {}
    
    print("\n=== Running Comprehensive Verilog Analysis ===")
    
    # 1. Basic Syntax Check
    outputs["syntax"], error = run_command(f"iverilog -tnull {design_file}", "Verilog Syntax Check")
    test_results.append("✅ Passed Syntax Check" if not error else f"❌ Failed Syntax Check\n{error}")
    
    # 2. Testbench Simulation
    outputs["compile"], error = run_command(f"iverilog -o {output_dir}/design_tb {design_file} {testbench_file}", "Compiling Testbench")
    if error:
        test_results.append(f"❌ Compilation Failed\n{error}")
        outputs["simulation"] = ""
    else:
        outputs["simulation"], error = run_command(f"vvp {output_dir}/design_tb", "Running Simulation")
        test_results.append(f"📝 Simulation Output:\n{outputs['simulation']}")
    
    # 3. Verilator Lint & Analysis
    outputs["lint"], error = run_command(f"verilator --lint-only {design_file}", "Verilator Lint Check")
    test_results.append("✅ Passed Verilator Lint Check" if not error else f"❌ Failed Verilator Lint Check\n{error}")
    
    # 4. Coverage Analysis (if applicable)
    if os.path.exists(f"{output_dir}/design_tb.vvp"):
        outputs["coverage"], _ = run_command(f"vvp -c {output_dir}/design_tb.vvp", "Coverage Analysis")
        test_results.append(f"📊 Coverage Analysis:\n{outputs['coverage']}")
    
    # 5. Yosys Synthesis & Optimization
    outputs["yosys"], error = run_command(
        f"echo 'read_verilog {design_file}; synth -top design; opt -full; stat' | yosys", 
        "Yosys Synthesis", 
        timeout=120
    )
    test_results.append("✅ Passed Yosys Synthesis" if not error else f"❌ Failed Yosys Synthesis\n{error}")
    
    # 6. Timing Analysis (simplified)
    outputs["timing"], _ = run_command(
        f"echo 'read_verilog {design_file}; synth; sta' | yosys", 
        "Static Timing Analysis"
    )
    test_results.append(f"⏱️ Timing Analysis:\n{outputs['timing']}")
    
    # Save all results to file
    result_text = "\n".join(test_results)
    with open(f"{output_dir}/analysis_results.txt", "w", encoding="utf-8") as f:
        f.write(result_text)
    
    # Extract metrics from outputs
    all_output = "\n".join([v for v in outputs.values() if v])
    metrics = extract_metrics_from_output(all_output)
    
    # Save metrics
    with open(f"{output_dir}/design_metrics.txt", "w", encoding="utf-8") as f:
        f.write(str(metrics))
    
    print(f"\n✅ Analysis completed! Results saved in '{output_dir}'.")
    return result_text, metrics


def create_design_template(user_request, output_dir):
    """Create design template files to guide the AI agents"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create design template
    with open(f"{output_dir}/design_template.v", "w") as f:
        f.write(f"""// Verilog Design Template
// Description: Implementation for "{user_request}"
// Author: Verilog AI Design System
// Date: {time.strftime("%Y-%m-%d")}

module design (
    // TODO: Define ports here
    input wire clk,
    input wire rst_n,
    // Additional ports will be defined based on requirements
);

    // Parameter definitions
    
    // Internal signals
    
    // Design implementation
    
    // TODO: Implement the design logic here
    
endmodule
""")
    
    # Create testbench template
    with open(f"{output_dir}/testbench_template.v", "w") as f:
        f.write(f"""// Verilog Testbench Template
// Description: Testbench for "{user_request}"
// Author: Verilog AI Design System
// Date: {time.strftime("%Y-%m-%d")}

`timescale 1ns/1ps

module testbench;
    // Test parameters
    
    // Testbench signals
    reg clk;
    reg rst_n;
    // Additional signals as needed
    
    // Device Under Test (DUT) instantiation
    design dut (
        .clk(clk),
        .rst_n(rst_n)
        // Connect additional ports
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end
    
    // Test stimulus and verification
    initial begin
        // Initialize signals
        rst_n = 0;
        
        // Apply reset
        #20 rst_n = 1;
        
        // TODO: Add test cases
        
        // End simulation
        #1000;
        $display("Tests completed!");
        $finish;
    end
    
    // Add assertions and checks
    
endmodule
""")

    return f"{output_dir}/design_template.v", f"{output_dir}/testbench_template.v"


def chat_with_agent(agent, user_prompt, groupchat):
    """Interact with an agent and return the extracted Verilog code."""
    groupchat.messages.append({"role": "user", "name": "User_proxy", "content": user_prompt})
    agent_text = agent.generate_reply(messages=groupchat.messages, user_input=user_prompt)
    
    if not isinstance(agent_text, str):
        agent_text = str(agent_text)
        
    groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
    verilog_code = extract_verilog_code(agent_text)
    
    return verilog_code, agent_text


# Main Design Process - Enhanced with multi-stage refinement and optimization
def orchestrate_design_flow(user_request, agents, groupchat, output_dir="./outputs"):
    """Advanced design process with architectural exploration and optimization"""
    # Create output directory structure
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/iterations").mkdir(exist_ok=True)
    Path(f"{output_dir}/optimization").mkdir(exist_ok=True)
    Path(f"{output_dir}/verification").mkdir(exist_ok=True)
    
    # Create design templates
    design_template, testbench_template = create_design_template(user_request, output_dir)
    
    # Initialize log and files
    conversation_log = []
    print("\n🚀 Starting Advanced Verilog Design Process...")
    
    # Step 1: Requirements Elaboration with Architect
    print("\n📋 Elaborating requirements...")
    requirements_elaboration, architect_req_response = chat_with_agent(
        agents["architect"],
        f"Analyze this design request and create a detailed requirements document: \"{user_request}\"\n\n"
        "Include functional requirements, performance targets, interfaces, corner cases, and design constraints.",
        groupchat
    )
    conversation_log.append(f"=== Requirements Analysis ===\n{architect_req_response}\n")
    
    # Step 2: Architecture Exploration - Consider 2-3 approaches
    print("\n🏛️ Exploring architecture options...")
    architecture_options, architect_options_response = chat_with_agent(
        agents["architect"],
        f"Propose 2-3 different architectural approaches for this design: \"{user_request}\"\n\n"
        f"Requirements overview:\n{requirements_elaboration}\n\n"
        "For each approach, describe the architecture, list pros/cons, and identify optimal use cases. "
        "These should be genuinely different approaches, not minor variations.",
        groupchat
    )
    conversation_log.append(f"=== Architecture Options ===\n{architect_options_response}\n")
    
    # Step 3: Architecture Selection & High-Level Design
    print("\n🏗️ Selecting optimal architecture and creating high-level design...")
    architecture_selection, architect_selection_response = chat_with_agent(
        agents["architect"],
        f"Based on the requirements and architectural options we've discussed, select the most appropriate architecture "
        f"for this design: \"{user_request}\"\n\n"
        "Explain your selection criteria and provide a detailed high-level design with module hierarchy, interfaces, "
        "and key design elements. Include a block diagram described in text.",
        groupchat
    )
    conversation_log.append(f"=== Architecture Selection ===\n{architect_selection_response}\n")
    
    # Step 4: Detailed Design Specification
    print("\n📝 Creating detailed design specification...")
    design_spec, architect_spec_response = chat_with_agent(
        agents["architect"],
        f"Create a detailed design specification based on the selected architecture for: \"{user_request}\"\n\n"
        "Include module interfaces (all ports with width and direction), internal signals, state machines, timing diagrams, "
        "algorithms, and design constraints. This will serve as the blueprint for implementation.",
        groupchat
    )
    conversation_log.append(f"=== Design Specification ===\n{architect_spec_response}\n")
    
    # Step 5: Initial Implementation by Coder
    print("\n💻 Implementing initial Verilog design...")
    initial_design, coder_response = chat_with_agent(
        agents["coder"],
        f"Implement the Verilog design according to this specification:\n\n{design_spec}\n\n"
        "Create a complete, synthesizable, and well-commented implementation. "
        "Follow Verilog best practices for synchronous design, reset handling, and coding style.",
        groupchat
    )
    
    if not initial_design:
        print("❌ Error: Coder did not produce valid Verilog code.")
        return None, None, ["Error: No valid Verilog design produced"]
    
    conversation_log.append(f"=== Initial Implementation ===\n{initial_design}\n")
    
    # Save initial version
    with open(f"{output_dir}/iterations/design_v1.v", "w", encoding="utf-8") as f:
        f.write(initial_design)
    
    # Step 6: Initial Testbench Creation
    print("\n🧪 Creating comprehensive testbench...")
    initial_testbench, tester_response = chat_with_agent(
        agents["tester"],
        f"Create a comprehensive self-checking testbench for this Verilog design:\n\n{initial_design}\n\n"
        "Include automated stimulus generation, reference model or behavioral checks, assertions, and coverage monitoring. "
        "Make sure to test all functionality, corner cases, and error conditions. "
        "The testbench should explicitly report PASS/FAIL status at completion.",
        groupchat
    )
    
    if not initial_testbench:
        print("❌ Error: Tester did not produce valid testbench code.")
        initial_testbench = "// Empty testbench - generation failed"
    
    conversation_log.append(f"=== Initial Testbench ===\n{initial_testbench}\n")
    
    # Save initial testbench
    with open(f"{output_dir}/iterations/testbench_v1.v", "w", encoding="utf-8") as f:
        f.write(initial_testbench)
    
    # Step 7: Design Review by Critic
    print("\n🔍 Performing detailed design review...")
    design_review, critic_response = chat_with_agent(
        agents["critic"],
        f"Perform a detailed review of this Verilog design:\n\n{initial_design}\n\n"
        "Check for: coding standards compliance, synthesis issues, timing problems, potential simulation/synthesis mismatches, "
        "reset behavior, clock domain crossings, and any other design weaknesses. "
        "Provide specific line-by-line feedback with severity ratings.",
        groupchat
    )
    conversation_log.append(f"=== Design Review ===\n{critic_response}\n")
    
    # Step 8: Testbench Review 
    print("\n🔍 Reviewing testbench quality...")
    testbench_review, critic_tb_response = chat_with_agent(
        agents["critic"],
        f"Review this Verilog testbench:\n\n{initial_testbench}\n\n"
        "Check for: comprehensiveness, self-checking mechanisms, corner case coverage, stimulus quality, "
        "assertion effectiveness, and debug capabilities. Suggest specific improvements.",
        groupchat
    )
    conversation_log.append(f"=== Testbench Review ===\n{critic_tb_response}\n")
    
    # Step 9: Design Refinement based on Reviews
    print("\n✏️ Refining design based on review feedback...")
    refined_design, coder_refined_response = chat_with_agent(
        agents["coder"],
        f"Refine this Verilog design based on the review feedback:\n\n"
        f"Original design:\n{initial_design}\n\n"
        f"Review feedback:\n{design_review}\n\n"
        "Address all issues while maintaining the original architecture. Explain your changes.",
        groupchat
    )
    
    if refined_design:
        conversation_log.append(f"=== Refined Design ===\n{refined_design}\n")
        with open(f"{output_dir}/iterations/design_v2.v", "w", encoding="utf-8") as f:
            f.write(refined_design)
    else:
        refined_design = initial_design
        print("⚠️ Using initial design as refinement failed.")
    
    # Step 10: Testbench Refinement
    print("\n✏️ Refining testbench based on review feedback...")
    refined_testbench, tester_refined_response = chat_with_agent(
        agents["tester"],
        f"Refine this Verilog testbench based on the review feedback:\n\n"
        f"Original testbench:\n{initial_testbench}\n\n"
        f"Review feedback:\n{testbench_review}\n\n"
        "Address all issues while improving test coverage and verification quality. Explain your changes.",
        groupchat
    )
    
    if refined_testbench:
        conversation_log.append(f"=== Refined Testbench ===\n{refined_testbench}\n")
        with open(f"{output_dir}/iterations/testbench_v2.v", "w", encoding="utf-8") as f:
            f.write(refined_testbench)
    else:
        refined_testbench = initial_testbench
        print("⚠️ Using initial testbench as refinement failed.")
    
    # Save current stable versions
    with open(f"{output_dir}/design.v", "w", encoding="utf-8") as f:
        f.write(refined_design)
    with open(f"{output_dir}/design_tb.v", "w", encoding="utf-8") as f:
        f.write(refined_testbench)
    
    # Step 11: Run initial analysis
    print("\n🧪 Running initial verification and analysis...")
    analysis_results, metrics = run_verilog_design_tools(output_dir)
    conversation_log.append(f"=== Initial Analysis ===\n{analysis_results}\n")
    conversation_log.append(f"=== Initial Metrics ===\n{str(metrics)}\n")
    
    # Test Failure Remediation Loop
    test_failure_iterations = 0
    max_test_failure_iterations = 3  # Limit remediation attempts
    
    while not metrics.simulation_status and test_failure_iterations < max_test_failure_iterations:
        test_failure_iterations += 1
        print(f"\n🔄 Test failures detected! Remediation iteration {test_failure_iterations}...")
        
        # Extract failure details from simulation output
        failure_lines = []
        for line in analysis_results.split('\n'):
            if re.search(r"FAIL:|failed:|test\s+failed", line, re.IGNORECASE):
                failure_lines.append(line)
        
        failure_summary = "\n".join(failure_lines) if failure_lines else "Tests failed but no specific failure messages found."
        print(f"Failure details:\n{failure_summary}")
        
        # Step 11A: Get failure analysis from Critic
        print("\n🔎 Analyzing test failures...")
        failure_analysis, critic_failure_response = chat_with_agent(
            agents["critic"],
            f"The Verilog testbench has detected failures for this design:\n\n{refined_design}\n\n"
            f"Testbench:\n{refined_testbench}\n\n"
            f"Failure details:\n{failure_summary}\n\n"
            f"Full test output:\n{analysis_results}\n\n"
            "Analyze the cause of these failures and provide a detailed diagnosis. "
            "Identify specific bugs, timing issues, or design flaws that could cause these test failures.",
            groupchat
        )
        conversation_log.append(f"=== Test Failure Analysis {test_failure_iterations} ===\n{critic_failure_response}\n")
        
        # Step 11B: Fix design based on analysis
        print("\n🛠️ Fixing design issues...")
        fixed_design, coder_fix_response = chat_with_agent(
            agents["coder"],
            f"Fix this Verilog design to address the test failures:\n\n"
            f"Current design:\n{refined_design}\n\n"
            f"Failure analysis:\n{failure_analysis}\n\n"
            f"Test failures:\n{failure_summary}\n\n"
            "Make targeted changes to fix these specific issues while maintaining the overall architecture. "
            "Explain your changes and reasoning.",
            groupchat
        )
        
        if fixed_design:
            conversation_log.append(f"=== Design Fix {test_failure_iterations} ===\n{fixed_design}\n")
            with open(f"{output_dir}/iterations/design_fixed_v{test_failure_iterations}.v", "w", encoding="utf-8") as f:
                f.write(fixed_design)
            refined_design = fixed_design
        else:
            print("⚠️ Failed to generate a fixed design. Continuing with current version.")
        
        # Step 11C: Update testbench if needed
        testbench_update_needed = "testbench issue" in failure_analysis.lower() or "test issue" in failure_analysis.lower()
        if testbench_update_needed:
            print("\n🧪 Updating testbench...")
            fixed_testbench, tester_fix_response = chat_with_agent(
                agents["tester"],
                f"Fix this Verilog testbench to address the issues:\n\n"
                f"Current testbench:\n{refined_testbench}\n\n"
                f"Failure analysis:\n{failure_analysis}\n\n"
                f"Test failures:\n{failure_summary}\n\n"
                "Make targeted changes to fix any testbench issues. "
                "Ensure the testbench correctly verifies the design's functionality.",
                groupchat
            )
            
            if fixed_testbench:
                conversation_log.append(f"=== Testbench Fix {test_failure_iterations} ===\n{fixed_testbench}\n")
                with open(f"{output_dir}/iterations/testbench_fixed_v{test_failure_iterations}.v", "w", encoding="utf-8") as f:
                    f.write(fixed_testbench)
                refined_testbench = fixed_testbench
        
        # Save updated files
        with open(f"{output_dir}/design.v", "w", encoding="utf-8") as f:
            f.write(refined_design)
        with open(f"{output_dir}/design_tb.v", "w", encoding="utf-8") as f:
            f.write(refined_testbench)
        
        # Re-run verification to check if issues are fixed
        print("\n🧪 Re-running verification after fixes...")
        analysis_results, metrics = run_verilog_design_tools(output_dir)
        conversation_log.append(f"=== Verification After Fix {test_failure_iterations} ===\n{analysis_results}\n")
        
        if metrics.simulation_status:
            print(f"✅ Test failures successfully resolved on iteration {test_failure_iterations}!")
            break
        elif test_failure_iterations == max_test_failure_iterations:
            print("⚠️ Maximum test failure remediation iterations reached. Continuing with current design.")
    
    # Step 12: Formal Verification Properties
    print("\n🔒 Adding formal verification properties...")
    formal_properties, formal_response = chat_with_agent(
        agents["formal_verifier"],
        f"Create formal verification properties for this Verilog design:\n\n{refined_design}\n\n"
        "Define key invariants, safety properties, liveness properties, and cover points "
        "that would verify the correctness of this design. Use SVA or PSL syntax.",
        groupchat
    )
    
    if formal_properties:
        conversation_log.append(f"=== Formal Properties ===\n{formal_properties}\n")
        with open(f"{output_dir}/verification/formal_properties.sv", "w", encoding="utf-8") as f:
            f.write(formal_properties)
    
    # Step 13: Optimization Phase
    if metrics.error_count == 0:  # Only optimize if no errors
        print("\n⚡ Optimizing design for performance and area...")
        optimized_design, optimizer_response = chat_with_agent(
            agents["optimizer"],
            f"Optimize this Verilog design for better performance and resource utilization:\n\n{refined_design}\n\n"
            f"Current metrics:\n{str(metrics)}\n\n"
            f"Analysis results:\n{analysis_results}\n\n"
            "Apply techniques like pipelining, retiming, FSM optimization, and resource sharing as appropriate. "
            "Maintain the same functionality while improving efficiency. Explain your optimization strategy.",
            groupchat
        )
        
        if optimized_design:
            conversation_log.append(f"=== Optimized Design ===\n{optimized_design}\n")
            with open(f"{output_dir}/optimization/design_optimized.v", "w", encoding="utf-8") as f:
                f.write(optimized_design)
            
            # Update current design with optimized version
            with open(f"{output_dir}/design.v", "w", encoding="utf-8") as f:
                f.write(optimized_design)
            
            # Run analysis on optimized design
            print("\n🧪 Analyzing optimized design...")
            opt_results, opt_metrics = run_verilog_design_tools(output_dir)
            conversation_log.append(f"=== Optimization Analysis ===\n{opt_results}\n")
            conversation_log.append(f"=== Optimization Metrics ===\n{str(opt_metrics)}\n")
            
            # Use optimized design if it's better
            if opt_metrics.error_count <= metrics.error_count and opt_metrics.optimization_level >= metrics.optimization_level:
                refined_design = optimized_design
                metrics = opt_metrics
                print("✅ Using optimized design (better metrics)")
            else:
                print("⚠️ Reverting to pre-optimization design (better stability)")
                # Revert to previous stable version
                with open(f"{output_dir}/design.v", "w", encoding="utf-8") as f:
                    f.write(refined_design)
    
    # Step 14: Final Design Documentation
    print("\n📚 Generating design documentation...")
    documentation, architect_doc_response = chat_with_agent(
        agents["architect"],
        f"Create comprehensive documentation for this Verilog design:\n\n{refined_design}\n\n"
        "Include: design overview, architecture description, module interfaces, design decisions, "
        "limitations, usage examples, and verification results. This should serve as a complete "
        "technical reference for the design.",
        groupchat
    )
    
    if documentation:
        conversation_log.append(f"=== Design Documentation ===\n{documentation}\n")
        with open(f"{output_dir}/design_documentation.md", "w", encoding="utf-8") as f:
            f.write(documentation)
    
    # Step 15: Final Design Summary
    print("\n📊 Creating final design summary...")
    
    # Pre-process variables that might contain backslashes to avoid f-string issues
    if architecture_selection and len(architecture_selection) > 500:
        arch_summary = architecture_selection[:500] + "..."
    else:
        arch_summary = architecture_selection or "Not specified"
    
    # Count newlines safely outside the f-string
    lines_of_code = 0
    if refined_design:
        lines_of_code = refined_design.count('\n')
    
    # Count test cases safely
    test_cases = 0
    if refined_testbench:
        test_cases = refined_testbench.count('initial begin')
    
    # Determine recommendations
    if metrics.error_count == 0 and metrics.warning_count < 5:
        impl_recommendation = "Design is ready for implementation"
    else:
        impl_recommendation = "Design requires further refinement"
        
    if metrics.optimization_level < 7:
        perf_recommendation = "Performance optimization recommended"
    else:
        perf_recommendation = "Performance is good"
        
    if metrics.coverage_percent < 80:
        test_recommendation = "Consider adding more test coverage"
    else:
        test_recommendation = "Test coverage is good"
    
    # Build the summary string without problematic backslashes in f-string expressions
    design_summary = f"""# Verilog Design Summary for "{user_request}"

## Design Status
- Implementation Complete: {'Yes' if refined_design else 'No'}
- Testbench Complete: {'Yes' if refined_testbench else 'No'}
- Formal Properties: {'Yes' if formal_properties else 'No'}
- Documentation: {'Yes' if documentation else 'No'}

## Design Metrics
{str(metrics)}

## Implementation Details
- Architecture: {arch_summary}
- Modules: {refined_design.count('module') if refined_design else 0}
- Lines of Code: {lines_of_code}

## Verification Status
- Simulation: {'Passed' if metrics.simulation_status else 'Failed'}
- Test Cases: {test_cases}

## Recommendations
- {impl_recommendation}
- {perf_recommendation}
- {test_recommendation}
"""
    
    with open(f"{output_dir}/design_summary.md", "w", encoding="utf-8") as f:
        f.write(design_summary)
    
    print("\n✅ Design process completed!")
    print(f"Output files available in: {output_dir}")
    
    return refined_design, refined_testbench, conversation_log


# Main execution function - Enhanced with more options
def main(user_request, output_dir="./verilog_output", model="gpt-4o-mini"):
    """Main execution function with proper initialization and cleanup"""
    # Setup output directory
    Path(output_dir).mkdir(exist_ok=True)
    start_time = time.time()
    
    print(f"🚀 Starting Verilog design process for: {user_request}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🤖 Using model: {model}")
    
    try:
        # Initialize LLM config and agents
        llm_config = setup_llm_config(model)
        agents = create_agents(llm_config)
        groupchat, manager = setup_chat_environment(agents)
        
        # Run the enhanced design flow
        design, testbench, log = orchestrate_design_flow(user_request, agents, groupchat, output_dir)
        
        # Save conversation log
        with open(f"{output_dir}/conversation_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if design and testbench:
            print(f"\n✅ Design process completed successfully in {duration:.1f} seconds!")
            print(f"📂 Files saved in {output_dir}")
            
            # Create README file
            with open(f"{output_dir}/README.md", "w", encoding="utf-8") as f:
                f.write(f"""# Verilog Design Project: {user_request}

## Overview
This directory contains a complete Verilog design implementation generated by the AI Verilog Design System.

## Files
- `design.v`: The main Verilog design implementation
- `design_tb.v`: Comprehensive testbench for verification
- `design_documentation.md`: Complete technical documentation
- `design_summary.md`: Metrics and status summary
- `iterations/`: Directory containing design evolution
- `optimization/`: Directory containing optimization attempts
- `verification/`: Directory containing verification assets

## Usage
1. Run simulation: `iverilog -o sim design.v design_tb.v && vvp sim`
2. Synthesize: `echo 'read_verilog design.v; synth; opt; stat' | yosys`

## Generation Info
- Request: "{user_request}"
- Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
- Process duration: {duration:.1f} seconds
""")
            
            return True
        else:
            print("\n❌ Design process encountered errors.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error in execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Allow running as script or importing as module
if __name__ == "__main__":
    open("design_flow_results.txt", "w", encoding="utf-8").close()
    open("verilog_test_results.txt", "w", encoding="utf-8").close()
    open("design.v", "w", encoding="utf-8").close()
    open("design_tb.v", "w", encoding="utf-8").close()
    
    # Example usage
    user_request = "Design a 4-bit synchronous counter with enable, up/down control, and parallel load"
    main(user_request)