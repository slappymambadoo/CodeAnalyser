# CodeAnalyser

**AI-powered code analyzer that generates honest portfolio skill reports. No CV inflation. Just what the code actually shows.**

## What It Does

- Recursively scans code folders
- Excludes binaries, dependencies, and generated files
- Splits files into LLM context-aware chunks
- Analyzes each chunk with a local LLM
- Synthesizes results into `auto_gen_purpose.md` and `auto_gen_skills.md`
- Generates a global portfolio summary (Markdown + PDF)

## Why I Built It

I wanted an honest assessment of my own portfolio. Instead of guessing what skills my code demonstrates, I built a tool to read my code and tell me.

## Requirements

- Python 3.8+
- A local LLM server (llama.cpp, vLLM, Ollama, or OpenAI-compatible endpoint)

## Installation

```bash
git clone https://github.com/slappymambadoo/CodeAnalyser.git
cd CodeAnalyser
pip install -r requirements.txt

Quick Start
bash
# Analyze a folder
python agent_folder_analyze.py /path/to/your/code

# Force re-analysis (ignore existing reports)
python agent_folder_analyze.py /path/to/your/code --force

# Generate global summary with PDF
python Global_Analyzer.py /path/to/analyzed/root --pdf

Configuration
Edit the script to customize exclusions, LLM endpoint, and context window.

Output
auto_gen_purpose.md - What the code does

auto_gen_skills.md - Technologies and skills demonstrated

global_summary.md / .pdf - Unified portfolio report

Example Global Summary Output
"This portfolio belongs to a highly autonomous, full-stack AI Engineer who acts as a 'One-Person R&D Lab.' Capable of designing, building, and deploying complex enterprise-grade systems that leverage Generative AI while maintaining strict adherence to security and compliance."

License
MIT

Disclaimer
This tool analyzes code using an LLM. Quality of output depends on your LLM and the code being analyzed. Always review generated reports.