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

# Define results analyzer agent
results_analyzer = autogen.AssistantAgent(
    name="results_analyzer",
    llm_config={"config_list": config_list},
    system_message="""You are a professional Verilog simulation results analyst. 
When presented with simulation results, provide a detailed analysis in Chinese.
Your analysis should include:
1. Overall assessment of the counter's functionality
2. Detailed timing analysis of each state change
3. Verification of reset and enable functionality
4. Assessment of the test coverage and completeness
5. Recommendations for improvements

Your analysis MUST be written in Chinese as this is a requirement."""
)

# Define user proxy for analysis (without custom streaming)
user_proxy = autogen.UserProxyAgent(
    name="analysis_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": WORK_DIR,
        "use_docker": False
    }
)

# Run the analysis
def run_analysis():
    print("===== 正在執行結果分析代理 =====")
    
    # Check if simulation output exists
    output_file = os.path.join(WORK_DIR, "simulation_output.txt")
    if not os.path.exists(output_file):
        print(f"\n❌ 找不到模擬輸出文件: {output_file}")
        print("請先運行開發者代理。")
        return False
    
    # Read simulation output
    with open(output_file, "r") as f:
        simulation_output = f.read()
    
    # If the file is empty or too small, return an error
    if len(simulation_output) < 10:
        print(f"\n❌ 模擬輸出文件為空或太小: {output_file}")
        return False
    
    print(f"\n📄 正在分析以下模擬結果:\n")
    print("----------------------------------")
    print(simulation_output)
    print("----------------------------------")
    print("\n開始分析...\n")
    
    # Start a new chat for analysis
    user_proxy.initiate_chat(
        results_analyzer,
        message=f"""請分析以下 Verilog 4位元計數器的模擬結果：

```
{simulation_output}
```

請提供詳細的中文分析報告，評估計數器的功能，並分析重置（reset）、使能（enable）和計數功能是否正常工作。
您的分析報告必須使用中文撰寫。"""
    )
    
    # Print the chat history
    for msg in user_proxy.chat_messages[results_analyzer.name]:
        print(f"\n{msg['role'].upper()}:")
        print(f"{msg['content']}")
        print("-" * 50)
    
    # Save the analysis to a file
    chat_history = user_proxy.chat_messages[results_analyzer.name]
    if chat_history:
        analysis_text = chat_history[-1]["content"]
        analysis_file = os.path.join(WORK_DIR, "analysis_report.txt")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"\n✅ 分析報告已保存到 {analysis_file}")
        return True
    else:
        print("\n❌ 聊天歷史中找不到分析結果。")
        return False

if __name__ == "__main__":
    success = run_analysis()
    # Return exit code based on success
    sys.exit(0 if success else 1)