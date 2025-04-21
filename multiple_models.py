import os
import subprocess
import openai
import requests
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API_KEY from the environment
api_key = os.getenv("API_KEY")
hf_token = os.getenv("HF_TOKEN")

from openai import OpenAI

class GPT4oMiniWrapper:
    def __init__(self, api_key):
        self.name = "gpt-4o-mini"
        self.client = OpenAI(api_key=api_key)

    def generate_code(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Verilog hardware expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

class HuggingFaceWrapper:
    def __init__(self, model_name, token=None):
        self.name = model_name
        self.model_name = model_name
        self.token = token

    def generate_code(self, prompt):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model_name}",
            headers=headers,
            json={"inputs": prompt}
        )
        try:
            return response.json()[0]['generated_text']
        except:
            return "// Error: Could not generate code."


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

def run_systemverilog_tests(test_filename, ref_filename, model_name):
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
        f.write(f"\n=== Test Run #{test_run_number}, Model: {model_name}===\n")
        f.write("\n".join(test_results) + "\n")

    print("\n✅ All tests completed! Results saved in 'verilog_test_results.txt'.")
    return "\n".join(test_results)

def extract_verilog_code(response_text):
    """Extract Verilog code block from agent response."""
    match = re.search(r"```verilog\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    if "module" in response_text:
        return response_text.strip()
    return "Error: No Verilog code detected"

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

def solve_with_llms(prompt_file, test_filename, ref_filename, api_key, hf_token):
    """
    Attempts to solve a Verilog design problem using multiple LLMs.

    Parameters:
        prompt_file (str): Path to the file containing the design prompt.
        test_filename (str): Path to the testbench file.
        ref_filename (str, optional): Optional reference file for simulation.
        api_key (str, optional): OpenAI API key for GPT-4o-mini.
    """
    prompt = extract_prompt(prompt_file)
    if not prompt:
        return "❌ Prompt file not found or empty."

    # Define LLMs to use (OpenAI + free Hugging Face models)
    models = []

    if api_key:
        models.append(GPT4oMiniWrapper(api_key))
    
    models += [
    HuggingFaceWrapper("deepseek-ai/deepseek-coder-6.7b-instruct", token=hf_token),
    HuggingFaceWrapper("meta-llama/Llama-2-7b-chat-hf", token=hf_token),
    HuggingFaceWrapper("mistralai/Mistral-7B-Instruct-v0.1", token=hf_token),
]


    for model in models:
        print(f"\n🤖 Trying model: {model.name}")
        response = model.generate_code(prompt)
        print(f"\n📤 Raw output from {model.name}:\n{response}\n")
        verilog_code = extract_verilog_code(response)

        # Save generated code to file
        with open("TopModule.v", "w") as f:
            f.write(verilog_code)

        # Run testbench
        test_results = run_systemverilog_tests(test_filename, ref_filename, model.name)
        tests_passed, reason = evaluate_test_results(test_results)

        if tests_passed:
            print("\n🎉 Code passed simulation!")
            return f"✅ Passed with model: {model.name}"
        else:
            print("\n❌ Code failed simulation, trying next model.")

    return "❌ All models failed to generate correct code."

if __name__ == "__main__":
    prompt_file = "Prob122_kmap4_prompt.txt"
    test_filename = "Prob122_kmap4_test.sv"
    ref_filename = "Prob122_kmap4_ref.sv"
    solve_with_llms(prompt_file, test_filename, ref_filename, api_key, hf_token)