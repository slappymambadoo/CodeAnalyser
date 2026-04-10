#!/usr/bin/env python3
"""
Global Analyzer - Collects all auto_gen_*.md files and produces unified summary.
"""

import os
import sys
import argparse
import json
import time
import requests
import re
from weasyprint import HTML

LLM_URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "LOCAL"
MAX_OUTPUT_TOKENS = 16384
DEFAULT_MAX_CONTEXT = 262144
CONTEXT_SAFETY_FACTOR = 0.90

# Tokenizer (same as original)
print("[INIT] Loading tokenizer...", flush=True)
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)
    print("[INIT] Qwen tokenizer loaded", flush=True)
except:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print("[INIT] Fallback tokenizer loaded", flush=True)

def count_tokens(text):
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except:
        return len(text) // 4

def collect_analysis_files(root_path):
    """Walk and collect all auto_gen files."""
    purposes = []
    skills_list = []
    
    for root, _, _ in os.walk(root_path):
        purpose_file = os.path.join(root, "auto_gen_purpose.md")
        skills_file = os.path.join(root, "auto_gen_skills.md")
        
        if os.path.exists(purpose_file):
            with open(purpose_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                purposes.append({
                    "folder": root,
                    "purpose": content,
                    "tokens": count_tokens(content)
                })
                print(f"  ✓ {root} (purpose: {purposes[-1]['tokens']} tokens)")
        
        if os.path.exists(skills_file):
            with open(skills_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                skills_list.append({
                    "folder": root,
                    "skills": content,
                    "tokens": count_tokens(content)
                })
    
    return purposes, skills_list

def send_to_llm(prompt, url, model):
    """Send to LLM and return response."""
    messages = [
        {"role": "system", "content": "You are a technical portfolio analyzer."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=3600)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        return re.sub(r' ?think.*? ?/think ?', '', content, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"  ❌ LLM error: {e}")
        raise

def markdown_to_pdf(markdown_file, output_pdf_path):
    """Convert existing markdown file to PDF."""
    try:
        # Read the markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown_content
        
        # Convert markdown-style headers to HTML
        html_content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
        
        # Convert lists
        html_content = re.sub(r'^\- (.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', html_content, flags=re.DOTALL)
        
        # Convert paragraphs (lines not already in HTML tags)
        lines = html_content.split('\n')
        processed_lines = []
        for line in lines:
            if line.strip() and not re.match(r'^<h[1-3]|<ul|<li|<strong|<em', line.strip()):
                processed_lines.append(f'<p>{line}</p>')
            else:
                processed_lines.append(line)
        html_content = '\n'.join(processed_lines)
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 40px;
                    color: #333;
                }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; margin-top: 20px; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                ul, ol {{ margin: 15px 0; }}
                hr {{ border: none; border-top: 1px solid #ecf0f1; margin: 30px 0; }}
                .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ecf0f1; font-size: 12px; color: #95a5a6; text-align: center; }}
            </style>
        </head>
        <body>
        {html_content}
        <div class="footer">Generated by Agent Folder Analyzer</div>
        </body>
        </html>
        """
        
        HTML(string=full_html).write_pdf(output_pdf_path)
        return True
    except Exception as e:
        print(f"  ❌ PDF generation error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory to scan")
    parser.add_argument("--direction", type=str, help="Focus direction")
    parser.add_argument("--output", type=str, default="global_summary.md")
    parser.add_argument("--context", type=int, default=DEFAULT_MAX_CONTEXT, help="Max context tokens")
    parser.add_argument("--url", type=str, default=LLM_URL)
    parser.add_argument("--model", type=str, default="LOCAL")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF from existing markdown report")
    parser.add_argument("--from-markdown", type=str, help="Convert existing markdown file to PDF (skips LLM generation)")
    args = parser.parse_args()
    
    # Handle PDF-only conversion from existing markdown
    if args.from_markdown:
        print(f"\n📄 Converting {args.from_markdown} to PDF...")
        pdf_path = args.from_markdown.replace('.md', '.pdf')
        if markdown_to_pdf(args.from_markdown, pdf_path):
            print(f"✅ PDF generated: {pdf_path}")
        else:
            print(f"❌ PDF conversion failed")
            sys.exit(1)
        return
    
    # Original behavior - generate report from scans
    print(f"\n🌍 Scanning {args.root}...")
    purposes, skills = collect_analysis_files(args.root)
    
    if not purposes:
        print("❌ No auto_gen files found. Run agent_folder_analyze.py first.")
        sys.exit(1)
    
    # Calculate total tokens
    purpose_tokens = sum(p['tokens'] for p in purposes)
    skills_tokens = sum(s['tokens'] for s in skills)
    total_tokens = purpose_tokens + skills_tokens
    available_tokens = int(args.context * CONTEXT_SAFETY_FACTOR)
    
    print(f"\n📊 Found {len(purposes)} analyzed folders")
    print(f"   Purpose tokens: {purpose_tokens}")
    print(f"   Skills tokens: {skills_tokens}")
    print(f"   Total: {total_tokens}")
    print(f"   Available (90% of {args.context}): {available_tokens}")
    
    if total_tokens > available_tokens:
        print(f"\n⚠️ WARNING: Exceeds 90% of context window by {total_tokens - available_tokens} tokens")
        print(f"   Consider: reducing scan scope, increasing --context, or implementing chunking")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"   ✅ Within limits ({total_tokens}/{available_tokens} tokens)")
    
    print(f"\n🤖 Generating global summary...")
    
    # Build prompt
    purposes_text = "\n\n".join([f"## {p['folder']}\n{p['purpose']}" for p in purposes])
    skills_text = "\n\n".join([f"## {s['folder']}\n{s['skills']}" for s in skills])
    direction_text = f"\n\nDIRECTION: {args.direction}" if args.direction else ""
    
    prompt = f"""Analyze these {len(purposes)} projects:

=== PURPOSES ===
{purposes_text}

=== SKILLS ===
{skills_text}
{direction_text}

Produce a single report with:
1. OVERALL ASSESSMENT - What kind of developer/system
2. SKILLS MASTER LIST - All unique skills, categorized
3. PROJECT SUMMARY - Each folder's role
4. EXPERIENCE REPORT - Technical expertise demonstrated

Output in clear markdown."""

    prompt_tokens = count_tokens(prompt)
    print(f"   Prompt tokens: {prompt_tokens}")
    
    summary = send_to_llm(prompt, args.url, args.model)
    
    # Write markdown output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Global Technical Summary\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Folders:** {len(purposes)}\n")
        f.write(f"**Direction:** {args.direction or 'None'}\n")
        f.write(f"**Tokens:** {total_tokens} (input) + {count_tokens(summary)} (output)\n\n")
        f.write(summary)
    
    print(f"\n✅ {args.output}")
    
    # Generate PDF if requested (from the newly created markdown)
    if args.pdf:
        pdf_path = args.output.replace('.md', '.pdf')
        print(f"\n📄 Generating PDF: {pdf_path}...")
        if markdown_to_pdf(args.output, pdf_path):
            print(f"✅ PDF generated: {pdf_path}")
        else:
            print(f"⚠️ PDF generation failed. Install weasyprint: pip install weasyprint")
    
    print("🖖 Make it so.")

if __name__ == "__main__":
    main()