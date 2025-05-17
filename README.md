# LLM-for-Hardware-Design

<img src="https://img.shields.io/badge/M.Eng_Project-Cornell-B31B1B" alt="Cornell M.Eng Project Badge">
<img src="https://img.shields.io/badge/Hardware-RTL_Design-blue" alt="Hardware Design">
<img src="https://img.shields.io/badge/AI-LLM_Applications-green" alt="LLM Applications">

This project explores the application of Large Language Models (LLMs) in automating and enhancing digital hardware design workflows. It integrates terminal-based agent orchestration, High-Level Synthesis (HLS) design automation, and Verilog RTL generation.

## Directory Structure

- `aiterminal/`: Scripts for coordinating LLM agents to generate and validate hardware designs via terminal interfaces.
- `hls_version/`: High-Level Synthesis implementations, including design automation scripts, reports, and benchmark results.
- `verilog_version/`: Verilog RTL modules and testbenches, including versions generated or refined using LLM tools.

## Objectives

- Automate the hardware design workflow using LLMs, from architecture descriptions to code generation.
- Compare LLM-assisted Verilog with HLS-generated and manually written designs.
- Evaluate design quality in terms of functional correctness, performance, and hardware efficiency.

## Technologies Used

- Python
- Vivado HLS
- Verilog HDL
- GitHub Copilot, OpenAI API

## Background

This repository was developed as part of a Master of Engineering (M.Eng) final project focused on applying language models to hardware design automation.

## Related Work

This repository complements the [ECE 6775 Final Project Repository](https://github.com/pl644/ece6775-final-project), which focused on benchmarking LLM performance in hardware design tasks through API implementations for ECE6775 Lab01 and Lab02 assignments, along with LLM-assisted implementation for Lab04. While the ECE 6775 project quantitatively evaluated LLM capabilities on specific lab assignments, this repository expands upon those findings to develop a comprehensive framework for applying LLMs across the entire hardware design workflow.
