import os
import json
import subprocess
import autogen
import re
import argparse
import asyncio
import time
import tqdm
import difflib
from colorama import Fore, Style, init
from dotenv import load_dotenv
from collections import deque
import hashlib

# Initialize colorama for colored console output
init()

# ======================================================
# Configuration and Setup Functions
# ======================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verilog ALU Design Automation")
    parser.add_argument("--max-iterations", type=int, default=3, 
                        help="Maximum number of iterations for each phase")
    parser.add_argument("--confidence-threshold", type=int, default=70,
                        help="Confidence threshold for hybrid evaluation (0-100)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use for LLM")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Environment variable name for API key")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip running tests (useful for development)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save output files")
    parser.add_argument("--use-ai-evaluation", action="store_true", 
                        help="Always use AI for test evaluation")
    parser.add_argument("--max-rollback", type=int, default=2,
                        help="Maximum number of versions to keep for rollback")
    parser.add_argument("--error-focus", action="store_true",
                        help="Focus on critical errors first")
    parser.add_argument("--incremental-fixes", action="store_true",
                        help="Apply fixes incrementally rather than all at once")
    return parser.parse_args()

def load_config():
    """Load configuration from environment and command line"""
    load_dotenv()  # Load environment variables from .env file
    args = parse_arguments()
    
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print_error(f"API key not found in environment variable {args.api_key_env}")
        print_info("Please set your API key using: export OPENAI_API_KEY=your_api_key")
        exit(1)
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    config = {
        "model": args.model,
        "max_iterations": args.max_iterations,
        "confidence_threshold": args.confidence_threshold,
        "skip_tests": args.skip_tests,
        "api_key": api_key,
        "output_dir": args.output_dir,
        "use_ai_evaluation": args.use_ai_evaluation,
        "max_rollback": args.max_rollback,
        "error_focus": args.error_focus,
        "incremental_fixes": args.incremental_fixes
    }
    
    return config

def check_environment():
    """Check if required tools are installed and accessible"""
    required_tools = ["iverilog", "vvp", "verilator", "yosys"]
    missing_tools = []
    
    for tool in required_tools:
        try:
            result = subprocess.run(f"which {tool}", shell=True, text=True, capture_output=True)
            if result.returncode != 0:
                missing_tools.append(tool)
        except Exception:
            missing_tools.append(tool)
    
    if missing_tools:
        print_warning(f"The following required tools are missing: {', '.join(missing_tools)}")
        print_info("Some tests may fail. Please install these tools to ensure proper functionality.")
        return False
    return True

# ======================================================
# User Interface Functions
# ======================================================

def print_header(text):
    """Print formatted header"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}= {text}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠️ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message"""
    print(f"{Fore.BLUE}ℹ️ {text}{Style.RESET_ALL}")

def print_diff(old_text, new_text, context=3):
    """Print colored diff between two texts"""
    diff = difflib.unified_diff(
        old_text.splitlines(True),
        new_text.splitlines(True),
        n=context
    )
    
    for line in diff:
        if line.startswith('+'):
            print(f"{Fore.GREEN}{line.rstrip()}{Style.RESET_ALL}")
        elif line.startswith('-'):
            print(f"{Fore.RED}{line.rstrip()}{Style.RESET_ALL}")
        elif line.startswith('^'):
            print(f"{Fore.BLUE}{line.rstrip()}{Style.RESET_ALL}")
        elif line.startswith('@@'):
            print(f"{Fore.CYAN}{line.rstrip()}{Style.RESET_ALL}")
        else:
            print(line.rstrip())

def show_progress(total_steps):
    """Create and return a progress bar"""
    return tqdm.tqdm(total=total_steps, desc="Design Progress", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

# ======================================================
# Command Execution Functions
# ======================================================

def run_command_with_retry(command, max_retries=2, timeout=60):
    """Run shell commands with retry logic and return output/errors"""
    print_info(f"Running command: {command}")
    
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout)
            
            if result.stderr and len(result.stderr.strip()) > 0:
                print_warning(f"Command produced error output:\n{result.stderr}")
            
            if result.returncode == 0 or attempt == max_retries:
                return result.stdout, result.stderr
                
            print_warning(f"Command failed (exit code {result.returncode}), retrying ({attempt+1}/{max_retries})...")
            
        except subprocess.TimeoutExpired:
            if attempt == max_retries:
                print_error(f"Command timed out after {timeout} seconds")
                return f"Command timed out after {timeout} seconds", "Timeout"
            print_warning(f"Command timed out, retrying ({attempt+1}/{max_retries})...")
            
        except Exception as e:
            print_error(f"Exception occurred: {str(e)}")
            return str(e), None
            
        # Wait before retrying
        time.sleep(2)
        
    return "Max retries exceeded", "Failed"

# ======================================================
# Code Analysis and Manipulation Functions
# ======================================================

def calculate_code_hash(code_text):
    """Calculate a hash of the code to track changes"""
    return hashlib.md5(code_text.encode('utf-8')).hexdigest()

def calculate_code_complexity(code_text):
    """Calculate a simple code complexity metric"""
    # Count non-comment, non-empty lines
    code_lines = 0
    comment_lines = 0
    empty_lines = 0
    
    for line in code_text.splitlines():
        stripped = line.strip()
        if not stripped:
            empty_lines += 1
        elif stripped.startswith('//') or stripped.startswith('/*'):
            comment_lines += 1
        else:
            code_lines += 1
    
    # Count modules and always blocks
    modules = len(re.findall(r'module\s+\w+', code_text))
    always_blocks = len(re.findall(r'always\s*@', code_text))
    case_statements = len(re.findall(r'case\s*\(', code_text))
    
    complexity = {
        'code_lines': code_lines,
        'comment_lines': comment_lines,
        'empty_lines': empty_lines,
        'modules': modules,
        'always_blocks': always_blocks,
        'case_statements': case_statements,
        'code_to_comment_ratio': code_lines / (comment_lines + 1),  # Avoid division by zero
        'overall_score': code_lines + (modules * 5) + (always_blocks * 3) + (case_statements * 2)
    }
    
    return complexity

def find_module_names(verilog_code):
    """Extract module names from Verilog code"""
    modules = re.findall(r'module\s+(\w+)', verilog_code)
    return modules

def check_design_coherence(alu_code, testbench_code):
    """Check if testbench and ALU module names match"""
    alu_modules = find_module_names(alu_code)
    testbench_modules = find_module_names(testbench_code)
    
    # Check if ALU modules are referenced in testbench
    missing_modules = []
    for module in alu_modules:
        if module not in testbench_code:
            missing_modules.append(module)
    
    # Check port connections
    alu_ports = re.findall(r'(?:input|output|inout)\s+(?:reg|wire)?\s*(?:\[\d+:\d+\])?\s*(\w+)', alu_code)
    missing_ports = []
    
    for port in alu_ports:
        if port not in testbench_code:
            missing_ports.append(port)
    
    coherence = {
        'alu_modules': alu_modules,
        'testbench_modules': testbench_modules,
        'missing_modules': missing_modules,
        'missing_ports': missing_ports,
        'is_coherent': len(missing_modules) == 0 and len(missing_ports) == 0
    }
    
    return coherence

def analyze_verilog_code(verilog_code):
    """Deep static analysis of Verilog code structure"""
    analysis = {
        'syntax_errors': [],
        'potential_issues': [],
        'module_structure': {},
    }
    
    # Look for unmatched begin/end blocks
    begin_count = len(re.findall(r'\bbegin\b', verilog_code))
    end_count = len(re.findall(r'\bend\b', verilog_code))
    
    if begin_count != end_count:
        analysis['potential_issues'].append(
            f"Unbalanced begin/end blocks: {begin_count} begins vs {end_count} ends"
        )
    
    # Find module declarations and analyze them
    module_matches = re.finditer(r'module\s+(\w+)(?:\s*#\([^)]*\))?\s*\(([^;]*)\);(.*?)endmodule', 
                                 verilog_code, re.DOTALL)
    
    for match in module_matches:
        module_name = match.group(1)
        port_list = match.group(2)
        module_body = match.group(3)
        
        # Parse ports
        ports = []
        for port in port_list.split(','):
            port = port.strip()
            if port:
                ports.append(port)
        
        # Extract always blocks
        always_blocks = re.findall(r'always\s*@\s*\([^)]*\)(.*?)(?=always|endmodule|$)', 
                                  module_body, re.DOTALL)
        
        analysis['module_structure'][module_name] = {
            'ports': ports,
            'always_blocks': len(always_blocks),
            'assigns': len(re.findall(r'assign\s+', module_body)),
            'registers': len(re.findall(r'reg\s+', module_body)),
            'wires': len(re.findall(r'wire\s+', module_body)),
        }
    
    # Check for common Verilog issues
    if re.search(r'initial\s+begin', verilog_code) and not re.search(r'module\s+\w+_tb', verilog_code):
        analysis['potential_issues'].append(
            "Initial blocks found in non-testbench module (not synthesizable)"
        )
    
    if re.search(r'#\d+', verilog_code) and not re.search(r'module\s+\w+_tb', verilog_code):
        analysis['potential_issues'].append(
            "Delays found in non-testbench module (not synthesizable)"
        )
    
    return analysis

# ======================================================
# Test Evaluation Functions
# ======================================================

def evaluate_test_results(test_results):
    """
    Rule-based evaluation of Verilog test results to determine success/failure.
    
    Parameters:
    test_results (str): Output from the Verilog tests
    
    Returns:
    bool: True if tests passed, False if tests failed
    str: Detailed reason for the evaluation
    int: Confidence level (0-100) in the evaluation
    dict: Detailed analysis of test results
    """
    # Convert to lowercase for case-insensitive matching
    results_lower = test_results.lower()
    
    # Initialize detailed analysis
    analysis = {
        'positive_indicators': [],
        'negative_indicators': [],
        'test_counts': {
            'total': None,
            'passed': None,
            'failed': None
        },
        'match_patterns': {}
    }
    
    # Positive indicators - explicit pass statements
    positive_patterns = [
        "all tests passed",
        "passed: ",
        "total tests: ", "passed: ", "failed: 0",
        r"simulation finished",
        r"finished successfully",
        r"all\s+tests\s+passed",
        r"test\s+bench\s+completed",
        r"(\d+)/\1\s+tests\s+passed",
        r"total tests:\s*(\d+).*passed:\s*\1",
    ]
    
    # Negative indicators - explicit fail statements or error patterns
    negative_patterns = [
        "failed: [1-9]",
        "❌ failed",
        "error:",
        "exception:",
        "segmentation fault",
        "compilation failed",
        r"unknown\s+module",
        r"undefined\s+symbol",
        r"syntax\s+error",
        r"cannot\s+find\s+module",
        r"timescale\s+error",
        r"assertion\s+failure",
    ]
    
    # Count pattern matches
    for pattern in positive_patterns:
        matches = re.findall(pattern, results_lower, re.IGNORECASE)
        if matches:
            analysis['positive_indicators'].append((pattern, len(matches)))
            analysis['match_patterns'][pattern] = matches
    
    for pattern in negative_patterns:
        matches = re.findall(pattern, results_lower, re.IGNORECASE)
        if matches:
            analysis['negative_indicators'].append((pattern, len(matches)))
            analysis['match_patterns'][pattern] = matches
    
    # Look for fatal errors first
    fatal_errors = ["compilation failed", "segmentation fault"]
    for error in fatal_errors:
        if error in results_lower:
            return False, f"Fatal error detected: '{error}'", 95, analysis
    
    # Extract test counts if available
    failed_tests_pattern = re.search(r"failed:\s*(\d+)", results_lower)
    if failed_tests_pattern:
        analysis['test_counts']['failed'] = int(failed_tests_pattern.group(1))
    
    passed_tests_pattern = re.search(r"passed:\s*(\d+)", results_lower)
    if passed_tests_pattern:
        analysis['test_counts']['passed'] = int(passed_tests_pattern.group(1))
    
    total_tests_pattern = re.search(r"total tests:\s*(\d+)", results_lower)
    if total_tests_pattern:
        analysis['test_counts']['total'] = int(total_tests_pattern.group(1))
    
    # Logic for determining overall result
    if analysis['test_counts']['failed'] is not None and analysis['test_counts']['failed'] > 0:
        return False, f"Detected {analysis['test_counts']['failed']} failed tests in summary", 90, analysis
    
    if "all tests passed" in results_lower:
        return True, "Explicitly stated 'All tests PASSED'", 95, analysis
    
    if (analysis['test_counts']['passed'] is not None and 
        analysis['test_counts']['failed'] is not None and 
        analysis['test_counts']['failed'] == 0):
        return True, f"Summary shows {analysis['test_counts']['passed']} tests passed and 0 failed", 90, analysis
    
    # Count explicit failure markers
    fail_count = results_lower.count("❌ failed")
    if fail_count > 0:
        return False, f"Detected {fail_count} failure indicators", 80, analysis
    
    # Count explicit pass markers
    pass_count = results_lower.count("✅ passed")
    if pass_count > 0 and pass_count > fail_count:
        return True, f"More pass indicators ({pass_count}) than failure indicators ({fail_count})", 70, analysis
    
    # If we have more negative than positive indicators
    if len(analysis['negative_indicators']) > len(analysis['positive_indicators']):
        return False, "More negative indicators than positive ones", 65, analysis
    
    # If we have more positive than negative indicators
    if len(analysis['positive_indicators']) > len(analysis['negative_indicators']):
        return True, "More positive indicators than negative ones", 65, analysis
    
    # Default - if none of the above conditions are met, be conservative
    return False, "Could not confidently determine test results, defaulting to failure", 30, analysis


class AdaptiveTestEvaluator:
    """
    Enhanced test evaluator that improves over time by learning from previous evaluations
    """
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.evaluation_history = deque(maxlen=5)  # Keep last 5 evaluations
        self.error_patterns = {}  # Track recurring error patterns
    
    def add_to_history(self, result, test_output, evaluation, confidence):
        """Add evaluation to history with metadata"""
        # Extract key excerpts from test output
        key_excerpts = []
        
        # Look for error lines
        error_lines = re.findall(r'(?:error|warning|failed).*', test_output, re.IGNORECASE)
        key_excerpts.extend(error_lines[:3])  # Take first 3 errors
        
        # Look for test summary lines
        summary_lines = re.findall(r'(?:total|passed|failed).*tests.*', test_output, re.IGNORECASE)
        key_excerpts.extend(summary_lines)
        
        # Default excerpt if nothing found
        if not key_excerpts:
            key_excerpts = [test_output[:200]] if len(test_output) > 0 else ["<empty output>"]
        
        # Update error pattern tracking
        for line in error_lines:
            # Create a simplified pattern by removing specific details
            pattern = re.sub(r'\b\d+\b', 'N', line)  # Replace numbers with N
            pattern = re.sub(r'\b0x[0-9a-f]+\b', 'HEX', pattern)  # Replace hex with HEX
            pattern = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]+\b', 'ID', pattern)  # Replace identifiers with ID
            
            if pattern in self.error_patterns:
                self.error_patterns[pattern]['count'] += 1
                self.error_patterns[pattern]['last_seen'] = len(self.evaluation_history)
            else:
                self.error_patterns[pattern] = {
                    'count': 1, 
                    'last_seen': len(self.evaluation_history),
                    'example': line
                }
        
        # Add to history
        self.evaluation_history.append({
            'result': result,
            'excerpts': key_excerpts,
            'evaluation': evaluation,
            'confidence': confidence,
            'hash': hashlib.md5(test_output.encode()).hexdigest()[:10]
        })
    
    def analyze_recurring_patterns(self):
        """Analyze recurring error patterns"""
        recurring = []
        
        for pattern, data in self.error_patterns.items():
            if data['count'] > 1:
                recurring.append({
                    'pattern': pattern,
                    'count': data['count'],
                    'example': data['example'],
                    'recency': len(self.evaluation_history) - data['last_seen']
                })
        
        # Sort by count (most frequent first)
        recurring.sort(key=lambda x: (x['count'], -x['recency']), reverse=True)
        return recurring[:3]  # Return top 3 recurring patterns
    
    def evaluate(self, test_results):
        """Evaluate test results with AI, using history for context"""
        # First, perform rule-based evaluation to enhance context
        rule_result, rule_reason, rule_confidence, analysis = evaluate_test_results(test_results)
        
        # Create agents
        evaluator = autogen.AssistantAgent(
            name="Test_Evaluator",
            system_message=(
                "You are a Verilog test results evaluation expert. "
                "Your task is to analyze test output and determine if the tests passed successfully. "
                "Analyze the entire output, look for all indicators of success and failure, "
                "and provide a clear final judgment (PASS or FAIL) with detailed reasoning. "
                "Look specifically for patterns that indicate synthesis or simulation errors."
            ),
            llm_config=self.llm_config
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False}
        )
        
        # Prepare message with history context
        history_context = ""
        recurring_patterns = self.analyze_recurring_patterns()
        
        if recurring_patterns:
            history_context += "The following error patterns have occurred multiple times:\n"
            for i, pattern in enumerate(recurring_patterns):
                history_context += f"{i+1}. Occurred {pattern['count']} times - Example: {pattern['example']}\n"
        
        if self.evaluation_history:
            history_context += "\nHere are some previous evaluations for reference:\n\n"
            for i, entry in enumerate(self.evaluation_history):
                history_context += f"Example {i+1} (Hash: {entry['hash']}):\n"
                history_context += f"Output excerpts:\n"
                for excerpt in entry['excerpts']:
                    history_context += f"- {excerpt.strip()}\n"
                history_context += f"Evaluation: {entry['evaluation']}\n"
                history_context += f"Correct Result: {'PASS' if entry['result'] else 'FAIL'}\n\n"
        
        # Add rule-based analysis
        rule_analysis = "Rule-based analysis detected:\n"
        if analysis['positive_indicators']:
            rule_analysis += "Positive indicators:\n"
            for pattern, count in analysis['positive_indicators']:
                rule_analysis += f"- {pattern}: {count} occurrences\n"
        
        if analysis['negative_indicators']:
            rule_analysis += "Negative indicators:\n"
            for pattern, count in analysis['negative_indicators']:
                rule_analysis += f"- {pattern}: {count} occurrences\n"
        
        if any(v is not None for v in analysis['test_counts'].values()):
            rule_analysis += f"Test counts: "
            for k, v in analysis['test_counts'].items():
                if v is not None:
                    rule_analysis += f"{k}={v} "
        
        message = (
            "Below are the Verilog test output results. Please evaluate these results and determine if the tests passed. "
            "Provide your judgment ('PASS' or 'FAIL') and explain your reasoning in detail.\n\n"
            f"Test Results:\n```\n{test_results[:4000]}...\n```\n\n"  # Limit to 4000 chars for context length
            f"{rule_analysis}\n\n"
            f"{history_context}\n"
            "Start your response with 'TEST_RESULT: PASS' or 'TEST_RESULT: FAIL', then explain your reasoning. "
            "If there are any verilog errors or warnings, explain what they mean and how they might be fixed."
        )
        
        # Get evaluation
        user_proxy.initiate_chat(evaluator, message=message)
        response = user_proxy.chat_messages[evaluator.name][-1]["content"]
        
        # Parse result
        result_match = re.search(r"TEST_RESULT:\s*(PASS|FAIL)", response, re.IGNORECASE)
        if result_match:
            result = result_match.group(1).upper() == "PASS"
            explanation = response.split(result_match.group(0), 1)[1].strip()
            
            # Store in history for future reference
            self.add_to_history(result, test_results, explanation[:200], 
                              90 if result_match.group(0) else 70)
            
            return result, explanation, 90
        else:
            # Fallback logic
            if "pass" in response.lower() and "fail" not in response.lower():
                self.add_to_history(True, test_results, 
                                  "AI evaluation indicates tests passed, but no explicit marker.", 60)
                return True, "AI evaluation indicates tests passed, but no explicit marker.", 60
            else:
                self.add_to_history(False, test_results, 
                                  "AI evaluation did not provide clear result, conservatively judging as failure.", 60)
                return False, "AI evaluation did not provide clear result, conservatively judging as failure.", 60


def hybrid_test_evaluation(test_results, llm_config, confidence_threshold=70, adaptive_evaluator=None, force_ai=False):
    """
    Enhanced hybrid evaluation with more detailed analysis and error categorization
    
    Parameters:
    test_results (str): Output from the Verilog tests
    llm_config (dict): LLM configuration
    confidence_threshold (int): Confidence threshold below which to use AI (0-100)
    adaptive_evaluator (AdaptiveTestEvaluator, optional): Instance of adaptive evaluator
    force_ai (bool): If True, always use AI evaluation
    
    Returns:
    bool: True if tests passed, False if tests failed
    str: Detailed reason for the evaluation
    str: Method used for evaluation ("rules" or "ai")
    dict: Additional analysis data
    """
    # First try rule-based evaluation (unless we're forcing AI)
    details = {}
    
    if not force_ai:
        result, reason, confidence, analysis = evaluate_test_results(test_results)
        details = analysis
        
        # If confidence is high enough, return rule-based result
        if confidence >= confidence_threshold:
            return result, reason, "rules", details
    
    # Otherwise, use AI evaluation
    print_info("Using AI for test evaluation...")
    
    # Use adaptive evaluator if provided, otherwise do standard AI eval
    if adaptive_evaluator:
        ai_result, ai_reason, ai_confidence = adaptive_evaluator.evaluate(test_results)
        details['ai_confidence'] = ai_confidence
    else:
        # Create a specialized AI assistant for test evaluation
        evaluator = autogen.AssistantAgent(
            name="Test_Evaluator",
            system_message=(
                "You are a Verilog test results evaluation expert. "
                "Your task is to analyze test output and determine if the tests passed successfully. "
                "Analyze the entire output, look for all indicators of success and failure, "
                "and provide a clear final judgment (PASS or FAIL) with detailed reasoning."
            ),
            llm_config=llm_config
        )
        
        # Create a user proxy agent to receive evaluation results
        user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False}
        )
        
        # Prepare message containing the test results
        message = (
            "Below are the Verilog test output results. Please evaluate these results and determine if the tests passed. "
            "Provide your judgment ('PASS' or 'FAIL') and explain your reasoning.\n\n"
            f"Test Results:\n```\n{test_results[:4000]}...\n```\n\n"
            "Start your response with 'TEST_RESULT: PASS' or 'TEST_RESULT: FAIL', then explain your reasoning."
        )
        
        # Initialize conversation and get evaluation results
        user_proxy.initiate_chat(evaluator, message=message)
        
        # Extract results from AI response
        response = user_proxy.chat_messages[evaluator.name][-1]["content"]
        
        # Parse the AI's judgment
        result_match = re.search(r"TEST_RESULT:\s*(PASS|FAIL)", response, re.IGNORECASE)
        
        if result_match:
            ai_result = result_match.group(1).upper() == "PASS"
            
            # Extract explanation part
            ai_reason = response.split(result_match.group(0), 1)[1].strip()
            details['ai_response'] = ai_reason[:200] + "..."  # Store truncated response
            details['ai_confidence'] = 85
        else:
            # If no explicit result marker is found, make a conservative judgment
            if "pass" in response.lower() and "fail" not in response.lower():
                ai_result = True
                ai_reason = "AI evaluation indicates tests passed, but no explicit marker."
                details['ai_confidence'] = 60
            else:
                ai_result = False
                ai_reason = "AI evaluation did not provide clear result, conservatively judging as failure."
                details['ai_confidence'] = 60
            
            details['ai_response'] = response[:200] + "..."  # Store truncated response
    
    return ai_result, ai_reason, "ai", details


def analyze_test_failure(test_results):
    """
    Enhanced analysis of test failures with precise error categorization
    
    Returns:
    dict: Analysis results with categories of issues and suggested fixes
    """
    analysis = {
        'syntax_errors': [],
        'timing_issues': [],
        'assertion_failures': [],
        'simulation_errors': [],
        'connection_errors': [],
        'undefined_errors': [],
        'unknown_errors': [],
        'suggested_fixes': []
    }
    
    # Check for syntax errors
    syntax_patterns = [
        (r"syntax error[^.\n]*", "Syntax error found"),
        (r"unexpected token[^.\n]*", "Unexpected token in code"),
        (r"undefined symbol[^.\n]*", "Undefined symbol referenced"),
        (r"undeclared identifier[^.\n]*", "Undeclared identifier used"),
        (r"expecting [^,\n]*, found [^.\n]*", "Syntax error - wrong token")
    ]
    
    for pattern, label in syntax_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            # Get some context around the match
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 40)
            context = test_results[start:end]
            analysis['syntax_errors'].append({'context': context, 'type': label})
            
            # Add suggested fix based on error type
            if "unexpected token" in pattern:
                analysis['suggested_fixes'].append(
                    "Check for missing semicolons, brackets, or parentheses near the error"
                )
            elif "undefined symbol" in pattern or "undeclared identifier" in pattern:
                analysis['suggested_fixes'].append(
                    "Ensure all variables are properly declared before use"
                )
    
    # Check for timing issues
    timing_patterns = [
        (r"timing violation[^.\n]*", "Timing violation detected"),
        (r"race condition[^.\n]*", "Race condition detected"),
        (r"setup time violation[^.\n]*", "Setup time violation"),
        (r"hold time violation[^.\n]*", "Hold time violation")
    ]
    
    for pattern, label in timing_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 40)
            context = test_results[start:end]
            analysis['timing_issues'].append({'context': context, 'type': label})
            
            # Add suggested fix
            analysis['suggested_fixes'].append(
                "Review clock sensitivity lists and ensure proper signal synchronization"
            )
    
    # Check for assertion failures
    assertion_patterns = [
        (r"assertion failed[^.\n]*", "Assertion failure"),
        (r"assertion error[^.\n]*", "Assertion error"),
        (r"expected (.+) but got (.+)", "Value mismatch in test")
    ]
    
    for pattern, label in assertion_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 100)  # Get more context for assertions
            context = test_results[start:end]
            
            # For value mismatches, extract the expected and actual values
            if "expected" in pattern:
                expected = match.group(1) if match.groups() and len(match.groups()) > 0 else "unknown"
                actual = match.group(2) if match.groups() and len(match.groups()) > 1 else "unknown"
                analysis['assertion_failures'].append({
                    'context': context, 
                    'type': label,
                    'expected': expected,
                    'actual': actual
                })
                
                # Add specific fix for value mismatch
                analysis['suggested_fixes'].append(
                    f"Fix logic to output {expected} instead of {actual}"
                )
            else:
                analysis['assertion_failures'].append({'context': context, 'type': label})
    
    # Check for simulation errors
    simulation_patterns = [
        (r"segmentation fault", "Segmentation fault"),
        (r"memory access violation", "Memory access violation"),
        (r"stack overflow", "Stack overflow"),
        (r"simulation time limit reached", "Simulation timeout"),
        (r"(divide|division) by zero", "Division by zero")
    ]
    
    for pattern, label in simulation_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 40)
            context = test_results[start:end]
            analysis['simulation_errors'].append({'context': context, 'type': label})
            
            # Add suggested fix
            if "divide by zero" in pattern.lower():
                analysis['suggested_fixes'].append(
                    "Add checks to prevent division by zero conditions"
                )
            elif "segmentation fault" in pattern or "memory access" in pattern:
                analysis['suggested_fixes'].append(
                    "Check for out-of-bounds array access or null pointer issues"
                )
    
    # Check for connection/wiring errors
    connection_patterns = [
        (r"port '(\w+)' not found", "Port not found"),
        (r"cannot connect port '(\w+)'", "Cannot connect port"),
        (r"width mismatch.*port '(\w+)'", "Port width mismatch"),
        (r"module '(\w+)' not found", "Module not found")
    ]
    
    for pattern, label in connection_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 40)
            context = test_results[start:end]
            
            # Extract the port or module name
            name = match.group(1) if match.groups() else "unknown"
            analysis['connection_errors'].append({
                'context': context, 
                'type': label,
                'name': name
            })
            
            # Add specific fix
            if "not found" in label:
                analysis['suggested_fixes'].append(
                    f"Ensure {name} is correctly defined and spelled consistently"
                )
            elif "width mismatch" in label:
                analysis['suggested_fixes'].append(
                    f"Match the bit width of port {name} between module and instantiation"
                )
    
    # Check for undefined values
    undefined_patterns = [
        (r"undefined value.*signal '(\w+)'", "Undefined signal value"),
        (r"uninitialized (\w+)", "Uninitialized value"),
        (r"value 'x' detected", "X-value detected")
    ]
    
    for pattern, label in undefined_patterns:
        matches = re.finditer(pattern, test_results, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 40)
            end = min(len(test_results), match.end() + 40)
            context = test_results[start:end]
            
            # Extract the signal name if available
            name = match.group(1) if match.groups() else "unknown"
            analysis['undefined_errors'].append({
                'context': context, 
                'type': label,
                'name': name
            })
            
            # Add specific fix
            analysis['suggested_fixes'].append(
                f"Initialize {name} to a defined value before use"
            )
    
    # Deduplicate suggested fixes
    if analysis['suggested_fixes']:
        analysis['suggested_fixes'] = list(set(analysis['suggested_fixes']))
    
    # Any errors not categorized go to unknown_errors
    if not any(len(errors) > 0 for category, errors in analysis.items() 
              if category != 'suggested_fixes'):
        # Extract lines with "error" or "warning" for generic categorization
        error_lines = re.findall(r'.*(?:error|warning|failed).*', test_results, re.IGNORECASE)
        if error_lines:
            for line in error_lines[:5]:  # Take first 5 errors
                analysis['unknown_errors'].append({'context': line, 'type': 'Generic error'})
        else:
            analysis['unknown_errors'].append({
                'context': test_results[:200], 
                'type': 'Unclassified error'
            })
    
    return analysis

# ======================================================
# Verilog Testing Functions
# ======================================================

def run_verilog_tests(output_dir=".", save_intermediate=True):
    """
    Enhanced Verilog testing with more comprehensive steps and better output parsing
    """
    test_results = []
    detailed_results = {
        'syntax_check': {'status': None, 'output': None},
        'testbench': {'status': None, 'output': None},
        'lint_check': {'status': None, 'output': None},
        'optimization': {'status': None, 'output': None}
    }

    print_header("Running Enhanced Verilog Testing Suite")

    # Path to Verilog files
    alu_path = os.path.join(output_dir, "alu.v")
    tb_path = os.path.join(output_dir, "alu_tb.v")
    
    # Check if files exist
    if not os.path.exists(alu_path):
        print_error(f"ALU file not found at {alu_path}")
        return "ERROR: ALU file not found"
    
    if not os.path.exists(tb_path):
        print_error(f"Testbench file not found at {tb_path}")
        return "ERROR: Testbench file not found"

    # Step 1: Verilog Syntax Check (ALU)
    print_header("Verilog Syntax Check (ALU)")
    output, error = run_command_with_retry(f"iverilog -tnull {alu_path}")
    if error:
        print_error(f"Syntax check failed: {error}")
        test_results.append(f"❌ Failed ALU Syntax Check\n{error}")
        detailed_results['syntax_check'] = {'status': 'fail', 'output': error}
    else:
        print_success("ALU syntax check passed")
        test_results.append("✅ Passed ALU Syntax Check")
        detailed_results['syntax_check'] = {'status': 'pass', 'output': output}

    # Step 2: Verilog Syntax Check (Testbench)
    print_header("Verilog Syntax Check (Testbench)")
    output, error = run_command_with_retry(f"iverilog -tnull {tb_path}")
    if error:
        print_error(f"Testbench syntax check failed: {error}")
        test_results.append(f"❌ Failed Testbench Syntax Check\n{error}")
    else:
        print_success("Testbench syntax check passed")
        test_results.append("✅ Passed Testbench Syntax Check")

    # Step 3: Running Verilog Testbench
    print_header("Running Verilog Testbench")
    output, error = run_command_with_retry(f"iverilog -o {output_dir}/alu_tb {alu_path} {tb_path}")
    if error:
        print_error(f"Compilation failed: {error}")
        test_results.append(f"❌ Compilation Failed\n{error}")
        detailed_results['testbench'] = {'status': 'fail', 'output': error}
    else:
        print_success("Compilation successful, running simulation...")
        sim_output, sim_error = run_command_with_retry(f"vvp {output_dir}/alu_tb")
        
        full_output = sim_output
        if sim_error:
            full_output += f"\nERROR OUTPUT:\n{sim_error}"
            print_warning(f"Simulation produced errors: {sim_error}")
        
        test_results.append(f"📝 Simulation Output:\n{full_output}")
        
        # Determine success based on output
        if "All tests PASSED" in sim_output or "passed: " in sim_output and "failed: 0" in sim_output:
            detailed_results['testbench'] = {'status': 'pass', 'output': full_output}
        else:
            detailed_results['testbench'] = {'status': 'fail', 'output': full_output}

    # Step 4: Running Verilator Lint Check
    print_header("Running Verilator Lint Check")
    output, error = run_command_with_retry(f"verilator --lint-only {alu_path}")
    if error and "Error" in error:
        print_error(f"Verilator lint check failed: {error}")
        test_results.append(f"❌ Failed Verilator Lint Check\n{error}")
        detailed_results['lint_check'] = {'status': 'fail', 'output': error}
    else:
        # Verilator might output warnings but still pass
        status = "pass" if not error or "Error" not in error else "warn"
        print_success("Verilator lint check passed" if status == "pass" else "Verilator lint check produced warnings")
        test_results.append("✅ Passed Verilator Lint Check" if status == "pass" else f"⚠️ Verilator Lint Check Warnings\n{error}")
        detailed_results['lint_check'] = {'status': status, 'output': output if not error else error}

    # Step 5: Checking Optimization with Yosys
    print_header("Checking Optimization with Yosys")
    output, error = run_command_with_retry(f"echo 'read_verilog {alu_path}; synth' | yosys")
    if error and not "Warning" in error:
        print_error(f"Yosys optimization check failed: {error}")
        test_results.append(f"❌ Failed Yosys Optimization Check\n{error}")
        detailed_results['optimization'] = {'status': 'fail', 'output': error}
    else:
        # Check for synthesis success in output
        if "Executing SYNTH pass" in output and "ERROR" not in output.upper():
            print_success("Yosys optimization check passed")
            test_results.append("✅ Passed Yosys Optimization Check")
            detailed_results['optimization'] = {'status': 'pass', 'output': output}
        else:
            print_warning("Yosys produced warnings or encountered issues")
            test_results.append(f"⚠️ Yosys Optimization Warnings\n{output}")
            detailed_results['optimization'] = {'status': 'warn', 'output': output}

    # Save test results
    test_results_path = os.path.join(output_dir, "verilog_test_results.txt")
    test_run_number = 1
    if os.path.exists(test_results_path):
        with open(test_results_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
            test_run_number = sum(1 for line in existing_lines if "Test Run #" in line) + 1

    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== Test Run #{test_run_number} ===\n")
        f.write("\n".join(test_results) + "\n")
    
    # Save detailed results if requested
    if save_intermediate:
        detailed_path = os.path.join(output_dir, f"test_run_{test_run_number}_details.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)

    print_success(f"All tests completed! Results saved in '{test_results_path}'")
    return "\n".join(test_results)

# ======================================================
# Agent Interaction Functions
# ======================================================

def extract_verilog_code(response_text):
    """
    Enhanced Verilog code extraction with better fallback mechanisms
    
    Returns:
    str: Extracted Verilog code or error message
    """
    # Try to extract code from markdown code blocks
    markdown_match = re.search(r"```(?:verilog)?\n(.*?)```", response_text, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1)

    # Try to find module definition (for plain text responses)
    module_match = re.search(r"(module\s+\w+\s*(?:\#\([^)]*\))?\s*\([^;]*\);.*?endmodule)", response_text, re.DOTALL)
    if module_match:
        return module_match.group(1)
    
    # Look for specific Verilog patterns
    if "module" in response_text and "endmodule" in response_text:
        # Try to extract from beginning of module to end of endmodule
        module_start = response_text.find("module")
        if module_start >= 0:
            endmodule_end = response_text.rfind("endmodule") + len("endmodule")
            if endmodule_end > module_start:
                return response_text[module_start:endmodule_end]
    
    return "Error: No valid Verilog code detected"


def chat_with_agent(agent, user_prompt, groupchat, user_proxy, code_extract=True, retries=1):
    """
    Enhanced agent interaction with retry logic and better error handling
    
    Parameters:
    agent (autogen.Agent): Agent to interact with
    user_prompt (str): Prompt to send to the agent
    groupchat (autogen.GroupChat): GroupChat instance
    user_proxy (autogen.UserProxyAgent): User proxy agent
    code_extract (bool): Whether to extract code from response
    retries (int): Number of retry attempts if no code is found
    
    Returns:
    str: Extracted content from agent response
    dict: Additional metadata about the interaction
    """
    metadata = {
        'agent_name': agent.name,
        'response_length': 0,
        'error': None,
        'retry_count': 0
    }
    
    # Add prompt to group chat
    groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": user_prompt})

    # Try to get a valid response
    for attempt in range(retries + 1):
        if attempt > 0:
            metadata['retry_count'] = attempt
            print_warning(f"Retrying with {agent.name} (attempt {attempt}/{retries})...")
            
            # Add clarification for retry
            retry_prompt = (
                f"I didn't get valid Verilog code in your previous response. "
                f"Please provide the complete Verilog code in a code block using the ```verilog ... ``` format."
            )
            groupchat.messages.append({"role": "user", "name": user_proxy.name, "content": retry_prompt})
        
        try:
            # Generate reply from agent
            agent_text = agent.generate_reply(messages=groupchat.messages)
            agent_text = str(agent_text) if agent_text is not None else ""
            
            # Add agent response to group chat
            groupchat.messages.append({"role": "assistant", "name": agent.name, "content": agent_text})
            
            metadata['response_length'] = len(agent_text)
            
            # Extract content if requested
            if code_extract:
                extracted_code = extract_verilog_code(agent_text)
                if "Error:" in extracted_code and attempt < retries:
                    # Will retry
                    continue
                return extracted_code, metadata
            else:
                return agent_text, metadata
                
        except Exception as e:
            metadata['error'] = str(e)
            print_error(f"Error with {agent.name}: {str(e)}")
            return f"Error communicating with {agent.name}: {str(e)}", metadata
    
    # If we get here, all retries failed
    return "Error: Could not extract valid Verilog code after multiple attempts", metadata

# ======================================================
# Version Management
# ======================================================

class VersionManager:
    """
    Manages versions of code for tracking changes and enabling rollback
    """
    def __init__(self, max_versions=3):
        self.versions = {
            'alu': deque(maxlen=max_versions),
            'testbench': deque(maxlen=max_versions)
        }
        self.current = {
            'alu': None,
            'testbench': None
        }
        self.test_results = deque(maxlen=max_versions)
    
    def add_version(self, code_type, code, test_result=None, metadata=None):
        """Add a new version of code"""
        if code_type not in self.versions:
            raise ValueError(f"Invalid code type: {code_type}")
        
        # Create version info
        version_info = {
            'code': code,
            'timestamp': time.time(),
            'hash': calculate_code_hash(code),
            'complexity': calculate_code_complexity(code),
            'metadata': metadata or {}
        }
        
        # Add test result if provided
        if test_result is not None:
            self.test_results.append({
                'alu_hash': self.current['alu']['hash'] if self.current['alu'] else None,
                'testbench_hash': self.current['testbench']['hash'] if self.current['testbench'] else None,
                'result': test_result,
                'timestamp': time.time()
            })
        
        # Store version
        self.versions[code_type].append(version_info)
        self.current[code_type] = version_info
        
        return version_info
    
    def get_latest_version(self, code_type):
        """Get the latest version of a code type"""
        if not self.versions[code_type]:
            return None
        return self.versions[code_type][-1]
    
    def rollback(self, code_type):
        """Rollback to the previous version"""
        if len(self.versions[code_type]) <= 1:
            return None  # Nothing to roll back to
        
        # Remove latest version and return the new latest
        self.versions[code_type].pop()
        self.current[code_type] = self.versions[code_type][-1]
        return self.current[code_type]
    
    def compare_versions(self, code_type, index1=-1, index2=-2):
        """Compare two versions of code"""
        if len(self.versions[code_type]) < abs(min(index1, index2)):
            return None
        
        version1 = self.versions[code_type][index1]
        version2 = self.versions[code_type][index2]
        
        # Compare code
        diff = difflib.unified_diff(
            version1['code'].splitlines(),
            version2['code'].splitlines(),
            fromfile=f"Version {index1}",
            tofile=f"Version {index2}",
            lineterm=''
        )
        
        # Compare complexity
        complexity_diff = {}
        for key in version1['complexity']:
            if key in version2['complexity']:
                old_val = version1['complexity'][key]
                new_val = version2['complexity'][key]
                complexity_diff[key] = {
                    'old': old_val,
                    'new': new_val,
                    'change': new_val - old_val,
                    'percent': ((new_val - old_val) / old_val * 100) if old_val != 0 else float('inf')
                }
        
        return {
            'diff': list(diff),
            'complexity_diff': complexity_diff,
            'time_difference': version2['timestamp'] - version1['timestamp'],
        }
    
    def should_rollback(self, test_results):
        """Decide whether to rollback based on test results"""
        if len(self.test_results) < 2:
            return False
        
        current = self.test_results[-1]
        previous = self.test_results[-2]
        
        # Simple heuristic: If current failed and previous passed
        if not current['result'] and previous['result']:
            return True
        
        # More complex heuristic: Count error patterns
        current_errors = analyze_test_failure(test_results)
        
        # If there are more errors now, consider rolling back
        current_error_count = sum(len(errors) for category, errors in current_errors.items()
                              if category != 'suggested_fixes')
        
        # We don't have the previous error count stored, but we can use the test result
        # as a proxy - if the previous test passed, rolling back might be good
        if current_error_count > 0 and previous['result']:
            return True
        
        return False
    
    def get_version_history(self, code_type):
        """Get a summary of version history"""
        if not self.versions[code_type]:
            return []
        
        history = []
        for i, version in enumerate(self.versions[code_type]):
            history.append({
                'version': i+1,
                'timestamp': version['timestamp'],
                'hash': version['hash'][:8],
                'lines': version['complexity']['code_lines'] + version['complexity']['comment_lines'],
                'complexity_score': version['complexity']['overall_score']
            })
        
        return history

# ======================================================
# Fix Prioritization
# ======================================================

def prioritize_fixes(error_analysis, code_analysis):
    """
    Intelligently prioritize fixes based on error severity and code structure
    
    Returns:
    list: Prioritized list of fixes to apply
    """
    fixes = []
    
    # 1. First, handle any critical/fatal errors
    critical_categories = ['syntax_errors', 'simulation_errors']
    for category in critical_categories:
        if category in error_analysis and error_analysis[category]:
            for error in error_analysis[category]:
                fixes.append({
                    'category': category,
                    'priority': 'critical',
                    'error': error
                })
    
    # 2. Then address connection/module structure issues
    if 'connection_errors' in error_analysis:
        for error in error_analysis['connection_errors']:
            fixes.append({
                'category': 'connection_errors',
                'priority': 'high',
                'error': error
            })
    
    # 3. Then handle assertion failures (functional issues)
    if 'assertion_failures' in error_analysis:
        for error in error_analysis['assertion_failures']:
            fixes.append({
                'category': 'assertion_failures',
                'priority': 'medium',
                'error': error
            })
    
    # 4. Then address timing issues
    if 'timing_issues' in error_analysis:
        for error in error_analysis['timing_issues']:
            fixes.append({
                'category': 'timing_issues',
                'priority': 'medium',
                'error': error
            })
    
    # 5. Then undefined value issues
    if 'undefined_errors' in error_analysis:
        for error in error_analysis['undefined_errors']:
            fixes.append({
                'category': 'undefined_errors',
                'priority': 'low',
                'error': error
            })
    
    # 6. Finally any other unknown errors
    if 'unknown_errors' in error_analysis:
        for error in error_analysis['unknown_errors']:
            fixes.append({
                'category': 'unknown_errors',
                'priority': 'low',
                'error': error
            })
    
    # Sort by priority
    priority_map = {
        'critical': 0,
        'high': 1,
        'medium': 2,
        'low': 3
    }
    
    fixes.sort(key=lambda x: priority_map[x['priority']])
    return fixes

def generate_fix_prompt(error_priority_list, full_error_analysis, code_analysis, incremental=False):
    """
    Generate a detailed prompt for fixing errors based on prioritized list
    
    Parameters:
    error_priority_list (list): Prioritized list of errors
    full_error_analysis (dict): Full error analysis
    code_analysis (dict): Code analysis
    incremental (bool): Whether to focus on incremental fixes
    
    Returns:
    str: Prompt for the agent
    """
    prompt = "Based on the test results, the following issues need to be addressed:\n\n"
    
    # Determine how many fixes to include based on incremental flag
    max_fixes = 1 if incremental else len(error_priority_list)
    fixes_to_include = error_priority_list[:max_fixes]
    
    for i, fix in enumerate(fixes_to_include):
        prompt += f"{i+1}. {fix['priority'].upper()} PRIORITY: {fix['category']}\n"
        prompt += f"   Context: {fix['error']['context']}\n"
        if 'type' in fix['error']:
            prompt += f"   Type: {fix['error']['type']}\n"
        prompt += "\n"
    
    # Add analysis of code structure
    if code_analysis:
        prompt += "Code analysis reveals the following structure:\n"
        if 'module_structure' in code_analysis:
            for module, details in code_analysis['module_structure'].items():
                prompt += f"Module '{module}':\n"
                prompt += f"- {len(details['ports'])} ports\n"
                prompt += f"- {details['always_blocks']} always blocks\n"
                prompt += f"- {details['assigns']} assign statements\n"
                prompt += f"- {details['registers']} registers\n"
                prompt += f"- {details['wires']} wires\n"
        
        if 'potential_issues' in code_analysis and code_analysis['potential_issues']:
            prompt += "\nPotential code issues:\n"
            for issue in code_analysis['potential_issues']:
                prompt += f"- {issue}\n"
    
    # Add suggested fixes from error analysis
    if 'suggested_fixes' in full_error_analysis and full_error_analysis['suggested_fixes']:
        prompt += "\nSuggested fixes:\n"
        for fix in full_error_analysis['suggested_fixes']:
            prompt += f"- {fix}\n"
    
    # Add instructions
    prompt += "\n"
    if incremental:
        prompt += "Please focus ONLY on fixing the FIRST issue listed above. Do not make any other changes to the code.\n"
    else:
        prompt += "Please fix ALL of the issues listed above while maintaining the original functionality and structure.\n"
    
    prompt += "Make sure your revised code:\n"
    prompt += "1. Is complete and correct Verilog syntax\n"
    prompt += "2. Preserves the overall functionality of the original design\n"
    prompt += "3. Has appropriate comments explaining the changes\n"
    prompt += "4. Will not introduce new issues\n"
    
    return prompt

# ======================================================
# Main Design Flow Class
# ======================================================

class VerilogDesignPipeline:
    """
    Complete Verilog design and testing pipeline with improved error handling and evaluation
    """
    def __init__(self, config):
        """Initialize the pipeline with configuration"""
        self.config = config
        self.setup_llm_config()
        self.create_agents()
        self.conversation_log = []
        self.progress = None
        
        # Initialize test evaluator
        self.adaptive_evaluator = AdaptiveTestEvaluator(self.llm_config)
        
        # Initialize version manager
        self.version_manager = VersionManager(max_versions=config['max_rollback'])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize result variables
        self.alu_design = None
        self.testbench = None
        self.fix_history = []
        
    def setup_llm_config(self):
        """Set up LLM configuration"""
        config_list = [
            {
                'model': self.config['model'],
                'api_key': self.config['api_key'],
            },
        ]
        os.environ["OAI_CONFIG_LIST"] = json.dumps(config_list)
        os.environ["AUTOGEN_USE_DOCKER"] = "0"
        self.llm_config = {"config_list": config_list, "cache_seed": 42}
    
    def create_agents(self):
        """Create all necessary agents with improved system messages"""
        # User proxy agent
        self.user_proxy = autogen.UserProxyAgent(
            name="User_proxy",
            system_message="A human engineer interacting with the Verilog team.",
            human_input_mode="TERMINATE",
            code_execution_config={"use_docker": False}
        )

        # Architect agent
        self.architect = autogen.AssistantAgent(
            name="Architect",
            system_message=(
                "You are an expert in Verilog architecture design. Provide efficient and modular designs. "
                "Your specialty is creating clean, synthesizable, and robust RTL that follows best practices. "
                "Explain your design choices clearly with comments when necessary. "
                "Be careful about signal widths, port connections, and synchronous logic design. "
                "Favor using explicit port connections and minimize behavioral code that isn't synthesizable."
            ),
            llm_config=self.llm_config
        )

        # Coder agent
        self.coder = autogen.AssistantAgent(
            name="Coder",
            system_message=(
                "You are a Verilog coding expert. Write clean, efficient, and well-commented Verilog code. "
                "Your specialty is implementing designs with high quality RTL. Follow these best practices: "
                "1. Use non-blocking assignments (<=) in sequential logic and blocking (=) in combinational "
                "2. Always define all signals and parameters with appropriate widths "
                "3. Use meaningful signal names and add documentation comments "
                "4. Ensure your always blocks have complete sensitivity lists "
                "5. Make designs synthesizable by avoiding initial blocks, delays, etc. in synthesizable modules "
                "When making fixes, consider impact on timing and potential for introducing new issues."
            ),
            llm_config=self.llm_config
        )

        # Critic agent
        self.critic = autogen.AssistantAgent(
            name="Critic",
            system_message=(
                "You are responsible for reviewing Verilog code for syntax correctness, logical consistency, "
                "and efficiency improvements. You are exceptionally good at spotting: "
                "1. Mismatched port connections between modules "
                "2. Signal width mismatches in assignments "
                "3. Incomplete sensitivity lists in always blocks "
                "4. Potential timing issues and race conditions "
                "5. Simulation vs. synthesis mismatches "
                "Provide constructive feedback while preserving the original code structure."
            ),
            llm_config=self.llm_config
        )

        # Tester agent
        self.tester = autogen.AssistantAgent(
            name="Tester",
            system_message=(
                "You are responsible for generating Verilog testbenches to validate functionality. "
                "Create comprehensive tests that: "
                "1. Verify all operations of the design under test "
                "2. Include edge cases and error conditions "
                "3. Provide clear pass/fail indicators for each test "
                "4. Are self-checking using assertions or comparisons "
                "5. Output an overall PASS/FAIL summary with counts "
                "Your testbenches should be thorough and clearly indicate test results."
            ),
            llm_config=self.llm_config
        )

        # Test coder agent
        self.test_coder = autogen.AssistantAgent(
            name="Test_Coder",
            system_message=(
                "You are responsible for improving Verilog testbenches by refining assertions, coverage, "
                "and self-checking mechanisms. Your specialty is: "
                "1. Adding comprehensive test cases for edge conditions "
                "2. Ensuring proper clock generation and timing in tests "
                "3. Creating exhaustive test scenarios for the DUT "
                "4. Adding clear reporting of test results with pass/fail counts "
                "5. Automating verification to catch all potential issues "
                "Keep the original structure intact but improve reliability and coverage."
            ),
            llm_config=self.llm_config
        )

        # Test critic agent
        self.test_critic = autogen.AssistantAgent(
            name="Test_Critic",
            system_message=(
                "You are responsible for reviewing Verilog testbenches for coverage, correctness, and effectiveness. "
                "Look specifically for: "
                "1. Missing test cases or edge conditions "
                "2. Improper clock/timing generation "
                "3. Unclear test result reporting "
                "4. Poor test organization or readability "
                "5. Lack of self-checking capabilities "
                "Provide feedback on missing cases, logical errors, and automation quality."
            ),
            llm_config=self.llm_config
        )

        # Fix specialist agent
        self.fix_specialist = autogen.AssistantAgent(
            name="Fix_Specialist",
            system_message=(
                "You are an expert in fixing Verilog code issues. Your specialty is analyzing test failures and "
                "error messages to propose targeted fixes. You are especially good at: "
                "1. Interpreting simulator/synthesis error messages "
                "2. Diagnosing testbench failures "
                "3. Fixing port connection issues between modules "
                "4. Resolving timing issues and race conditions "
                "5. Correcting signal width mismatches "
                "When fixing code, make minimal changes required to resolve the issue. "
                "Document your changes with comments explaining what was fixed and why."
            ),
            llm_config=self.llm_config
        )

        # Create GroupChat
        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.architect, self.coder, self.critic, 
                    self.tester, self.test_coder, self.test_critic, self.fix_specialist],
            messages=[],
            max_round=12
        )
        
        # Create GroupChatManager
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)
    
    def log_step(self, step_name, content):
        """Add a step to the conversation log"""
        self.conversation_log.append(f"=== {step_name} ===\n{content}\n")
        
    def save_conversation_log(self):
        """Save the conversation log to a file"""
        log_path = os.path.join(self.config['output_dir'], "design_flow_results.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.conversation_log))
        print_success(f"Conversation log saved to {log_path}")
    
    def save_verilog_files(self, alu_code, testbench_code):
        """Save Verilog files to output directory"""
        alu_path = os.path.join(self.config['output_dir'], "alu.v")
        tb_path = os.path.join(self.config['output_dir'], "alu_tb.v")
        
        with open(alu_path, "w", encoding="utf-8") as f:
            f.write(alu_code)
            
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(testbench_code)
            
        print_success(f"ALU code saved to {alu_path}")
        print_success(f"Testbench code saved to {tb_path}")
    
    def run_design_flow(self, user_request):
        """
        Run the complete ALU design flow
        
        Parameters:
        user_request (str): User's initial request
        
        Returns:
        tuple: (alu_code, testbench_code, conversation_log)
        """
        total_steps = 6  # Total major steps in the pipeline
        self.progress = show_progress(total_steps)
        
        print_header("Starting Verilog ALU Design Process")
        
        # Step 1: Generate initial architecture
        self.progress.set_description("Generating Architecture Design")
        self.generate_architecture()
        self.progress.update(1)
        
        # Step 2: Generate initial testbench
        self.progress.set_description("Generating Testbench")
        self.generate_testbench()
        self.progress.update(1)
        
        # Step 3: Refine ALU design
        self.progress.set_description("Refining ALU Design")
        self.refine_alu_design()
        self.progress.update(1)
        
        # Step 4: Refine testbench
        self.progress.set_description("Refining Testbench")
        self.refine_testbench()
        self.progress.update(1)
        
        # Step 5: Save files and run tests
        self.progress.set_description("Running Tests")
        self.save_verilog_files(self.alu_design, self.testbench)
        
        # Skip tests if requested
        if self.config['skip_tests']:
            print_info("Skipping tests as requested")
            test_results = "Tests skipped by user request"
        else:
            test_results = run_verilog_tests(self.config['output_dir'])
        self.progress.update(1)
        
        # Step 6: Fix issues if needed
        self.progress.set_description("Fixing Issues")
        success = self.test_and_fix(test_results)
        self.progress.update(1)
        
        # Final step: Save log
        self.save_conversation_log()
        self.progress.close()
        
        if success:
            print_header("🎉 Design Process Completed Successfully! 🎉")
        else:
            print_warning("Design process completed with potential issues.")
            
        return self.alu_design, self.testbench, self.conversation_log
    
    def generate_architecture(self):
        """Generate initial architecture design with improved prompting"""
        print_header("Generating Initial ALU Architecture")
        
        architect_prompt = (
            "Provide a Verilog design for a 4-bit ALU, including key functionalities and considerations. "
            "The ALU should support at least the following operations:\n"
            "1. ADD: A + B\n"
            "2. SUB: A - B\n"
            "3. AND: A & B\n"
            "4. OR: A | B\n"
            "5. NOT: ~A\n"
            "6. XOR: A ^ B\n"
            "\nInclude proper error handling, overflow detection, and make the design modular and efficient.\n\n"
            "Important design considerations:\n"
            "- Ensure all ports are explicitly defined with correct widths\n"
            "- Include thorough commenting explaining operation and signal purposes\n"
            "- Use proper Verilog coding practices for synthesis\n"
            "- Avoid potential timing issues with proper synchronous design\n"
            "- Include parameters for configurability\n"
            "\nProvide the complete module in a ```verilog code block."
        )
        
        print_info("Asking Architect for initial ALU design...")
        self.alu_design, metadata = chat_with_agent(
            self.architect,
            architect_prompt,
            self.groupchat,
            self.user_proxy
        )
        
        # Add to version manager
        self.version_manager.add_version('alu', self.alu_design, metadata=metadata)
        
        self.log_step("Architect's Design", self.alu_design)
        print_success("Architect provided an ALU design")
        
    def generate_testbench(self):
        """Generate initial testbench with improved prompting"""
        print_header("Generating Initial Testbench")
        
        tester_prompt = (
            f"This is the Architect's ALU design:\n\n```verilog\n{self.alu_design}\n```\n\n"
            "Generate a thorough Verilog testbench (using Verilog, not SystemVerilog) that verifies all "
            "ALU functionalities. Your testbench should:\n\n"
            "1. Exercise all operations (ADD, SUB, AND, OR, NOT, XOR)\n"
            "2. Test edge cases (overflow, zero results, etc.)\n"
            "3. Include self-checking mechanisms using if-else checks\n"
            "4. Count and report total tests, passes, and failures\n"
            "5. Display final PASS/FAIL status after all tests complete\n"
            "6. Have clear test organization with comments explaining each test case\n\n"
            "Be sure to match all signal names, ports, and parameter values exactly as defined in the ALU code. "
            "Ensure that at the end of the testbench, the output explicitly states whether the tests have "
            "PASSED or FAILED, including a count of total, passed, and failed tests.\n\n"
            "Provide the complete testbench in a ```verilog code block."
        )
        
        print_info("Asking Tester to generate testbench...")
        self.testbench, metadata = chat_with_agent(
            self.tester,
            tester_prompt,
            self.groupchat,
            self.user_proxy
        )
        
        # Add to version manager
        self.version_manager.add_version('testbench', self.testbench, metadata=metadata)
        
        self.log_step("Tester's Initial Testbench", self.testbench)
        print_success("Tester provided an initial testbench")
        
    def refine_alu_design(self):
        """Refine ALU design through multiple iterations with enhanced feedback"""
        print_header("Refining ALU Design")
        
        max_iterations = self.config['max_iterations']
        
        for i in range(max_iterations):
            print_info(f"ALU Design Iteration {i+1}/{max_iterations}")
            
            # Get code analysis first
            code_analysis = analyze_verilog_code(self.alu_design)
            
            # Get feedback from critic with more specific instructions
            critic_prompt = (
                f"Here is the current Verilog ALU code:\n\n```verilog\n{self.alu_design}\n```\n\n"
                "Please review this code thoroughly and identify any issues or improvements needed. "
                "Focus on the following aspects:\n"
                "1. Syntax correctness and Verilog best practices\n"
                "2. Signal width mismatches or port connection issues\n"
                "3. Potential timing issues or race conditions\n"
                "4. Incomplete sensitivity lists in always blocks\n"
                "5. Missing functionality compared to requirements\n"
                "6. Simulation vs. synthesis mismatches\n\n"
                "For each issue found, provide a clear explanation of:\n"
                "- What the issue is\n"
                "- Why it's a problem\n"
                "- How to fix it\n\n"
                "If no issues are found, state that explicitly."
            )
            
            print_info(f"Requesting critique for ALU design (Iteration {i+1})...")
            critic_response, critic_metadata = chat_with_agent(
                self.critic,
                critic_prompt,
                self.groupchat,
                self.user_proxy,
                code_extract=False
            )
            
            self.log_step(f"Critic Feedback (Iteration {i+1})", critic_response)
            
            # Check if no issues found
            if "no issues" in critic_response.lower() or "looks good" in critic_response.lower():
                print_success("ALU design finalized - no further issues found!")
                break
            
            # Refinement by coder with specific instructions
            coder_prompt = (
                f"Here is the current Verilog ALU code:\n\n```verilog\n{self.alu_design}\n```\n\n"
                f"Here is the feedback from the Critic:\n\n{critic_response}\n\n"
                "Please refine the Verilog code based on this feedback while keeping the original structure intact. "
                "Implement all suggested improvements and fix any issues mentioned. Make sure your revised code:\n"
                "1. Addresses ALL issues mentioned by the Critic\n"
                "2. Maintains the same interface (ports, parameters) for compatibility\n"
                "3. Includes comments explaining your changes\n"
                "4. Is complete and correct Verilog that could be synthesized\n\n"
                "Provide the complete revised Verilog code in a ```verilog code block."
            )
            
            print_info(f"Refining ALU design based on feedback (Iteration {i+1})...")
            refined_alu, coder_metadata = chat_with_agent(
                self.coder,
                coder_prompt,
                self.groupchat,
                self.user_proxy
            )
            
            # Check if meaningful changes were made
            if calculate_code_hash(refined_alu) == calculate_code_hash(self.alu_design):
                print_warning("No significant changes made to ALU design, stopping refinement")
                break
            
            # Store previous version for comparison
            previous_alu = self.alu_design
            self.alu_design = refined_alu
            
            # Add to version manager
            self.version_manager.add_version('alu', self.alu_design, metadata=coder_metadata)
            
            # Show diff
            print_info(f"Changes made in iteration {i+1}:")
            print_diff(previous_alu, self.alu_design)
            
            self.log_step(f"Coder (Version {i+1})", self.alu_design)
            print_success(f"ALU design updated (Version {i+1})")
    
    def refine_testbench(self):
        """Refine testbench through multiple iterations with better feedback"""
        print_header("Refining Testbench")
        
        max_iterations = self.config['max_iterations']
        
        for i in range(max_iterations):
            print_info(f"Testbench Iteration {i+1}/{max_iterations}")
            
            # Analyze testbench and check coherence with ALU
            coherence = check_design_coherence(self.alu_design, self.testbench)
            testbench_analysis = analyze_verilog_code(self.testbench)
            
            # Get feedback from test critic with specific instructions
            test_critic_prompt = (
                f"Here is the current Verilog ALU code:\n\n```verilog\n{self.alu_design}\n```\n\n"
                f"And here is the current testbench:\n\n```verilog\n{self.testbench}\n```\n\n"
                "Please review this testbench thoroughly and identify any issues or improvements needed. "
                "Focus on the following aspects:\n"
                "1. Proper module instantiation and port connections to the ALU\n"
                "2. Test coverage for all ALU operations and edge cases\n"
                "3. Self-checking mechanisms for test validation\n"
                "4. Clear reporting of test results with counts\n"
                "5. Overall testbench organization and readability\n\n"
            )
            
            # Add coherence check results if there are issues
            if not coherence['is_coherent']:
                test_critic_prompt += "There appear to be module/port connection issues:\n"
                if coherence['missing_modules']:
                    test_critic_prompt += f"- Missing modules: {', '.join(coherence['missing_modules'])}\n"
                if coherence['missing_ports']:
                    test_critic_prompt += f"- Missing ports: {', '.join(coherence['missing_ports'])}\n"
            
            test_critic_prompt += (
                "\nFor each issue found, provide a clear explanation of:\n"
                "- What the issue is\n"
                "- Why it's a problem\n"
                "- How to fix it\n\n"
                "If no issues are found, state that explicitly."
            )
            
            print_info(f"Requesting critique for testbench (Iteration {i+1})...")
            test_critic_response, critic_metadata = chat_with_agent(
                self.test_critic,
                test_critic_prompt,
                self.groupchat,
                self.user_proxy,
                code_extract=False
            )
            
            self.log_step(f"Test Critic Feedback (Iteration {i+1})", test_critic_response)
            
            # Check if no issues found
            if "no issues" in test_critic_response.lower() or "looks good" in test_critic_response.lower():
                print_success("Testbench finalized - no further issues found!")
                break
            
            # Refinement by test coder with specific instructions
            test_coder_prompt = (
                f"Here is the current Verilog ALU code:\n\n```verilog\n{self.alu_design}\n```\n\n"
                f"Here is the current testbench:\n\n```verilog\n{self.testbench}\n```\n\n"
                f"Here is the feedback from the Test Critic:\n\n{test_critic_response}\n\n"
                "Please refine the testbench based on this feedback while maintaining its structure. "
                "Implement all suggested improvements and fix any issues mentioned. Make sure your revised testbench:\n"
                "1. Addresses ALL issues mentioned by the Test Critic\n"
                "2. Correctly connects to the ALU module with exact port matches\n"
                "3. Tests all ALU operations thoroughly\n"
                "4. Includes clear test result reporting\n"
                "5. Is complete and correct Verilog that can be simulated\n\n"
                "Provide the complete revised testbench in a ```verilog code block."
            )
            
            print_info(f"Refining testbench based on feedback (Iteration {i+1})...")
            refined_testbench, coder_metadata = chat_with_agent(
                self.test_coder,
                test_coder_prompt,
                self.groupchat,
                self.user_proxy
            )
            
            # Check if meaningful changes were made
            if calculate_code_hash(refined_testbench) == calculate_code_hash(self.testbench):
                print_warning("No significant changes made to testbench, stopping refinement")
                break
            
            # Store previous version for comparison
            previous_testbench = self.testbench
            self.testbench = refined_testbench
            
            # Add to version manager
            self.version_manager.add_version('testbench', self.testbench, metadata=coder_metadata)
            
            # Show diff
            print_info(f"Changes made in iteration {i+1}:")
            print_diff(previous_testbench, self.testbench)
            
            self.log_step(f"Test Coder (Version {i+1})", self.testbench)
            print_success(f"Testbench updated (Version {i+1})")
    
    def test_and_fix(self, test_results):
        """Enhanced test and fix process with improved error handling and rollback"""
        print_header("Testing and Fixing Issues")
        
        max_iterations = self.config['max_iterations']
        
        for i in range(max_iterations):
            # Evaluate test results using hybrid approach
            tests_passed, reason, method, details = hybrid_test_evaluation(
                test_results, 
                self.llm_config,
                self.config['confidence_threshold'],
                self.adaptive_evaluator,
                self.config['use_ai_evaluation']
            )
            
            self.log_step(
                f"Test Evaluation (Iteration {i+1})",
                f"Method: {method}\nResult: {'PASS' if tests_passed else 'FAIL'}\nReason: {reason}"
            )
            
            # Store result in version manager
            self.version_manager.add_version(
                'alu', 
                self.alu_design, 
                test_result=tests_passed
            )
            
            if tests_passed:
                print_success(f"Tests PASSED! ({method} evaluation: {reason})")
                print_success("ALU and Testbench finalized!")
                return True
                
            print_warning(f"Tests FAILED! ({method} evaluation: {reason})")
            
            # Check if we should rollback
            if i > 0 and self.version_manager.should_rollback(test_results):
                print_warning("Test results have worsened, rolling back to previous version...")
                
                # Rollback ALU
                previous_alu = self.version_manager.rollback('alu')
                if previous_alu:
                    print_info("Rolled back ALU design to previous version")
                    self.alu_design = previous_alu['code']
                
                # Rollback testbench
                previous_tb = self.version_manager.rollback('testbench')
                if previous_tb:
                    print_info("Rolled back testbench to previous version")
                    self.testbench = previous_tb['code']
                
                # Save rolled back files
                self.save_verilog_files(self.alu_design, self.testbench)
                
                # Re-run tests with rolled back version
                test_results = run_verilog_tests(self.config['output_dir'])
                continue
            
            # Analyze test failures
            failure_analysis = analyze_test_failure(test_results)
            
            # Get code analysis
            alu_analysis = analyze_verilog_code(self.alu_design)
            tb_analysis = analyze_verilog_code(self.testbench)
            
            # Check design coherence
            coherence = check_design_coherence(self.alu_design, self.testbench)
            
            # Prioritize fixes
            alu_fixes = prioritize_fixes(failure_analysis, alu_analysis)
            
            # Generate fix prompts based on priority
            if self.config['error_focus']:
                # Only focus on one issue at a time
                incremental = True
            else:
                # Try to fix all issues at once
                incremental = False
                
            if not coherence['is_coherent']:
                # If there are coherence issues, prioritize fixing those
                print_warning("Coherence issues detected between ALU and testbench")
                fix_prompt = (
                    f"There are module/port connection issues between the ALU and testbench:\n\n"
                    f"ALU Code:\n```verilog\n{self.alu_design}\n```\n\n"
                    f"Testbench Code:\n```verilog\n{self.testbench}\n```\n\n"
                    f"Issues detected:\n"
                )
                
                if coherence['missing_modules']:
                    fix_prompt += f"- Missing modules: {', '.join(coherence['missing_modules'])}\n"
                if coherence['missing_ports']:
                    fix_prompt += f"- Missing ports: {', '.join(coherence['missing_ports'])}\n"
                
                fix_prompt += (
                    "\nPlease fix BOTH the ALU and testbench to ensure they are compatible. "
                    "Ensure that port names, widths, and connections match exactly between the two files. "
                    "Return both the fixed ALU and the fixed testbench code."
                )
                
                print_info("Requesting coherence fixes...")
                
                # Use Fix Specialist for coherence issues
                fix_specialist_response, metadata = chat_with_agent(
                    self.fix_specialist,
                    fix_prompt,
                    self.groupchat,
                    self.user_proxy,
                    code_extract=False
                )
                
                # Try to extract both ALU and testbench code
                alu_match = re.search(r"```verilog\s*(?:\/\/\s*ALU\s*Code)?.*?(module\s+\w+.*?endmodule)", 
                                     fix_specialist_response, re.DOTALL)
                tb_match = re.search(r"```verilog\s*(?:\/\/\s*Testbench\s*Code)?.*?(module\s+\w+_tb.*?endmodule)", 
                                    fix_specialist_response, re.DOTALL)
                
                if alu_match:
                    new_alu = alu_match.group(1)
                    self.alu_design = new_alu
                    print_success("Updated ALU design for coherence")
                    
                if tb_match:
                    new_tb = tb_match.group(1)
                    self.testbench = new_tb
                    print_success("Updated testbench for coherence")
                
                self.log_step(f"Coherence Fix (Iteration {i+1})", fix_specialist_response)
            else:
                # Generate fix prompt for ALU issues
                alu_fix_prompt = generate_fix_prompt(
                    alu_fixes, failure_analysis, alu_analysis, incremental
                )
                
                full_fix_prompt = (
                    f"The current ALU design has test failures. Here's the full context:\n\n"
                    f"Test Results:\n```\n{test_results[:2000]}\n```\n\n"
                    f"Current ALU code:\n```verilog\n{self.alu_design}\n```\n\n"
                    f"Issues to fix:\n{alu_fix_prompt}\n\n"
                    f"Please update the ALU code to fix these issues while maintaining its original functionality. "
                    f"Provide the complete revised ALU code in a ```verilog code block."
                )
                
                print_info("Requesting fixes for ALU issues...")
                fixed_alu, fix_metadata = chat_with_agent(
                    self.fix_specialist,
                    full_fix_prompt,
                    self.groupchat,
                    self.user_proxy
                )
                
                # Record the fix for history
                fix_record = {
                    'iteration': i+1,
                    'timestamp': time.time(),
                    'issues': [f['category'] for f in alu_fixes[:3]], # Top 3 issues
                    'old_hash': calculate_code_hash(self.alu_design),
                    'new_hash': calculate_code_hash(fixed_alu)
                }
                self.fix_history.append(fix_record)
                
                # Check if meaningful changes were made
                if calculate_code_hash(fixed_alu) == calculate_code_hash(self.alu_design):
                    print_warning("No significant changes made to ALU design, trying testbench fix")
                    
                    # Try fixing testbench instead
                    tb_fix_prompt = (
                        f"The current testbench may have issues. Here's the full context:\n\n"
                        f"Test Results:\n```\n{test_results[:2000]}\n```\n\n"
                        f"Current ALU code:\n```verilog\n{self.alu_design}\n```\n\n"
                        f"Current Testbench code:\n```verilog\n{self.testbench}\n```\n\n"
                        f"The tests are failing. Please analyze the test failures and update the testbench to "
                        f"improve test coverage or fix any issues in the testbench itself. "
                        f"Provide the complete revised testbench in a ```verilog code block."
                    )
                    
                    print_info("Requesting fixes for testbench...")
                    fixed_testbench, tb_fix_metadata = chat_with_agent(
                        self.test_coder,
                        tb_fix_prompt,
                        self.groupchat,
                        self.user_proxy
                    )
                    
                    if calculate_code_hash(fixed_testbench) != calculate_code_hash(self.testbench):
                        print_success("Updated testbench instead of ALU")
                        self.log_step(f"Testbench Fix (Iteration {i+1})", fixed_testbench)
                        self.testbench = fixed_testbench
                        self.version_manager.add_version('testbench', self.testbench, metadata=tb_fix_metadata)
                    else:
                        print_warning("No significant changes made to either ALU or testbench")
                else:
                    # Update ALU design
                    print_info("Comparing old and new ALU versions:")
                    print_diff(self.alu_design, fixed_alu)
                    
                    self.alu_design = fixed_alu
                    self.version_manager.add_version('alu', self.alu_design, metadata=fix_metadata)
                    self.log_step(f"ALU Fix (Iteration {i+1})", fixed_alu)
                    print_success(f"Updated ALU design with fixes (Iteration {i+1})")
            
            # Save updated files
            self.save_verilog_files(self.alu_design, self.testbench)
            
            # Re-run tests
            print_info("Re-running tests with updated code...")
            test_results = run_verilog_tests(self.config['output_dir'])
        
        # If we reach here, we've exhausted all iterations without success
        print_warning(f"Reached maximum iterations ({max_iterations}) without resolving all issues.")
        print_info("Here's a summary of the fix history:")
        
        for idx, fix in enumerate(self.fix_history):
            print(f"Fix #{idx+1} - Issues addressed: {', '.join(fix['issues'])}")
        
        return False

# ======================================================
# Main Execution
# ======================================================

def main():
    """Main execution function with enhanced setup and error handling"""
    try:
        # Check environment
        if not check_environment():
            choice = input("Some required tools are missing. Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                print_info("Exiting. Please install the required tools and try again.")
                return
        
        # Load configuration
        config = load_config()
        
        # Print configuration summary
        print_header("Verilog ALU Design Automation")
        print_info(f"Model: {config['model']}")
        print_info(f"Max iterations: {config['max_iterations']}")
        print_info(f"Confidence threshold: {config['confidence_threshold']}%")
        print_info(f"Output directory: {config['output_dir']}")
        print_info(f"Skip tests: {'Yes' if config['skip_tests'] else 'No'}")
        print_info(f"Force AI evaluation: {'Yes' if config['use_ai_evaluation'] else 'No'}")
        print_info(f"Max rollback versions: {config['max_rollback']}")
        print_info(f"Focus on critical errors: {'Yes' if config['error_focus'] else 'No'}")
        print_info(f"Apply incremental fixes: {'Yes' if config['incremental_fixes'] else 'No'}")
        
        # Initialize pipeline
        pipeline = VerilogDesignPipeline(config)
        
        # Run design flow
        user_request = "Design a 4-bit ALU in Verilog, write testbenches, and optimize the implementation."
        final_alu, final_testbench, _ = pipeline.run_design_flow(user_request)
        
        # Print summary
        print_header("Design Results")
        print_info(f"ALU code size: {len(final_alu.splitlines())} lines")
        print_info(f"Testbench code size: {len(final_testbench.splitlines())} lines")
        print_success(f"Files saved to: {config['output_dir']}")
        
        # Print complexity metrics
        alu_complexity = calculate_code_complexity(final_alu)
        print_header("ALU Complexity Metrics")
        print_info(f"Code lines: {alu_complexity['code_lines']}")
        print_info(f"Comment lines: {alu_complexity['comment_lines']}")
        print_info(f"Modules: {alu_complexity['modules']}")
        print_info(f"Always blocks: {alu_complexity['always_blocks']}")
        print_info(f"Case statements: {alu_complexity['case_statements']}")
        
        # Print version history
        print_header("Version History")
        alu_history = pipeline.version_manager.get_version_history('alu')
        print_info("ALU versions:")
        for version in alu_history:
            print(f"  Version {version['version']} - {time.strftime('%H:%M:%S', time.localtime(version['timestamp']))} - Hash: {version['hash']}")
        
        # Print fix history
        if pipeline.fix_history:
            print_header("Fix History")
            for idx, fix in enumerate(pipeline.fix_history):
                timestamp = time.strftime('%H:%M:%S', time.localtime(fix['timestamp']))
                print(f"Fix #{idx+1} ({timestamp}) - Issues: {', '.join(fix['issues'])}")
        
        # Print next steps
        print_header("Next Steps")
        print_info("You can now use the generated Verilog files in your project.")
        print_info(f"- ALU code: {os.path.join(config['output_dir'], 'alu.v')}")
        print_info(f"- Testbench: {os.path.join(config['output_dir'], 'alu_tb.v')}")
        print_info(f"- Design log: {os.path.join(config['output_dir'], 'design_flow_results.txt')}")
        print_info(f"- Test results: {os.path.join(config['output_dir'], 'verilog_test_results.txt')}")

    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    open("design_flow_results.txt", "w", encoding="utf-8").close()
    open("verilog_test_results.txt", "w", encoding="utf-8").close()

    
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())