#!/usr/bin/env python3
"""
Agent Folder Analyzer - Recursively analyzes folders using LLM
Excludes: known VCS/dependency folders (by name) + binary files (by content)
"""

import os
import sys
import requests
import argparse
import re
import fnmatch
import time
import json
from pathlib import Path
try:
    import pathspec
    HAS_PATHSPEC = True
except Exception:
    HAS_PATHSPEC = False

# ========== CONFIGURATION ==========
EXCLUDE_DIRS = {
    '.git', '.svn', '.hg', '__pycache__', '.venv', 'venv', 'env',
    'site-packages', 'dist-packages', 'lib', 'Lib', 'node_modules',
    'bin', 'obj', 'Debug', 'Release', 'net8.0', 'net6.0', 'net5.0',
    'netcoreapp3.1', 'netcoreapp2.0', 'ref', 'refint', 'runtimes',
    'target', 'build', 'out', '.gradle', '.alpackages', '.snapshots',
    '.vscode', '.idea', '.vs', 'dist', 'build-output',
    '.llm_pids',           # PID files directory
    '.pytest_cache',       # pytest cache
    '.ipynb_checkpoints',  # Jupyter checkpoints
    '.ruff_cache',         # Ruff cache
    '.llm_pids',
    '*:Zone.Identifier'
}

EXCLUDE_FILES = {
    '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes', '.gitmodules',
    'auto_gen_purpose.md', 'auto_gen_skills.md', 'auto_gen_requested_files.txt',
}

EXCLUDE_PATTERNS = {
    # Data files
    '*.csv', '*.xlsx', '*.xls', '*.parquet', '*.feather', '*.tsv',
    '*.json', '*.jsonl', '*.txt', '*.log', '*.bak', '*.tmp', '*.temp',
    '*.db', '*.sqlite', '*.7z', '*.zip', '*.tar', '*.gz',
    
    # Media files
    '*.mp3', '*.mp4', '*.m4a', '*.wav', '*.flac',
    '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff',
    
    # System files
    '*.pid', '*.lock', '*.user', '*.suo', '*.Zone.Identifier',
    '*.pyc', '*.pyo', '*.pdb',
    
    # Generated/Output files
    '*.kcb', '*.kcb.lock', '*.kdf-journal',
    '*.pdf', '*.epub', '*.mobi',
    
    # Large generated text files
    'story_Output.txt', 'horror_analysis.md', 'unique_outcomes.txt',
    'modelEvaluations_*.txt', 'filtered_medical_outcomes.txt',
    'prompt_*.md',                    # Generated prompt files
    '*_consensus_summary.txt',        # Summary files
    'analysis_*.md',                  # Analysis outputs
    'results.md',                     # Result files
    
    # Generated HTML patterns (reports, analysis outputs)
    'report_details/*.html',
    'research_reports/*.html',
    'risk_*.html',
    'assessment_*.html',
    'debug/*.html',
    'email_*.html',
    '*_analysis.html',
    '*_results.html',
    'output/*.html',
    'temp/*.html',
    'tmp/*.html',
    '*_debug_tables.html',
    '*_modifiers_table.html',
}

# Patterns for HTML that should be KEPT (application code)
KEEP_HTML_PATTERNS = {
    '*/app/*.html',
    '*/public/*.html',
    '*/frontend/*.html',
    '*/templates/*.html',
    '*/website/*.html',
    'index.html',
    '*.sln',  # Solution files
}

LLM_URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "LOCAL"
DEFAULT_MAX_CONTEXT = 262144
CONTEXT_SAFETY_FACTOR = 0.80
MAX_OUTPUT_TOKENS = 16384
FORCE_REANALYZE = False
DRY_RUN = False  # If True, only collect files and write list, no LLM calls

# ========== TOKENIZER ==========
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

# ========== EXCLUSION LOGIC ==========
def should_exclude_by_pattern(filename):
    """Check if filename matches any exclusion pattern."""
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

def should_keep_html(filepath):
    """Check if HTML file should be kept (application code) or excluded (generated report)."""
    # First check if it matches exclusion patterns
    if should_exclude_by_pattern(filepath):
        return False
    
    # Check if it's in a keep pattern
    for pattern in KEEP_HTML_PATTERNS:
        if fnmatch.fnmatch(filepath, pattern):
            return True
    
    # For other HTML files, check size (generated reports are often large)
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100 * 1024:  # >100KB
            return False
    except:
        pass
    
    # Default: exclude HTML unless explicitly kept
    return False

def should_exclude_folder(folder):
    """Exclude by name (known non-code) OR by content (virtual env signatures)."""
    if not os.path.exists("excluded.md"):
        open("excluded.md", "w").close()
    if not os.path.exists("included.md"):
        open("included.md", "w").close()
    
    folder_name = os.path.basename(folder)
    if folder_name in EXCLUDE_DIRS:
        reason = f"name match ({folder_name})"
        print(f"  [EXCLUDE] {folder} - {reason}", flush=True)
        with open("excluded.md", "a", encoding="utf-8") as f:
            f.write(f"{folder}\t# excluded: {reason}\n")
        return True
    
    if os.path.exists(os.path.join(folder, 'pyvenv.cfg')):
        reason = "pyvenv.cfg present"
        print(f"  [EXCLUDE] {folder} - {reason}", flush=True)
        with open("excluded.md", "a", encoding="utf-8") as f:
            f.write(f"{folder}\t# excluded: {reason}\n")
        return True
    
    try:
        for item in os.listdir(folder):
            if item.endswith('.pyc'):
                reason = "contains .pyc files"
                print(f"  [EXCLUDE] {folder} - {reason}", flush=True)
                with open("excluded.md", "a", encoding="utf-8") as f:
                    f.write(f"{folder}\t# excluded: {reason}\n")
                return True
    except:
        pass
    
    return False

def is_binary_file(filepath):
    """Check if file is binary."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
            try:
                chunk.decode('utf-8')
                return False
            except:
                return True
    except:
        return True

def load_gitignore_patterns(gitignore_path):
    """Load and return gitignore patterns, also print them for debugging."""
    lines = []
    print(f"\n  📄 Reading .gitignore: {gitignore_path}")
    try:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"     Raw content:\n{content}")
            for line in content.split('\n'):
                raw = line.rstrip('\n').strip()
                # Skip empty lines and comments
                if not raw or raw.startswith('#'):
                    continue
                lines.append(raw)
                print(f"     Pattern added: '{raw}'")
    except Exception as e:
        print(f"     Error reading: {e}")
        return None

    if not lines:
        print(f"     No patterns found")
        return None

    if HAS_PATHSPEC:
        try:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', lines)
            print(f"     Compiled {len(lines)} patterns with pathspec")
            return spec
        except Exception as e:
            print(f"     Pathspec compile error: {e}, using raw patterns")
            return lines
    return lines

def gitignore_matches(name, is_dir, patterns):
    """Return True if an entry name matches any pattern."""
    if not patterns:
        return False

    # Convert to posix for consistent matching
    name_posix = name.replace('\\', '/')
    
    # If it's a directory, add trailing slash for matching
    match_name = name_posix + '/' if is_dir else name_posix

    # If we have a compiled PathSpec, use it
    if HAS_PATHSPEC and hasattr(patterns, 'match_file'):
        # For directories, try with and without trailing slash
        if is_dir:
            return patterns.match_file(name_posix) or patterns.match_file(name_posix + '/')
        return patterns.match_file(name_posix)

    # Fallback: simple pattern matching
    for pat in patterns:
        pat_posix = pat.replace('\\', '/')
        
        # Handle directory patterns (ending with /)
        if pat_posix.endswith('/'):
            pat_dir = pat_posix.rstrip('/')
            # Match exact directory name or any subpath
            if name_posix == pat_dir or name_posix.startswith(pat_dir + '/'):
                return True
        else:
            # Simple glob pattern matching
            if fnmatch.fnmatch(name_posix, pat_posix):
                return True
            # Also try matching with **/ pattern (recursive)
            if '**' in pat_posix:
                pattern_parts = pat_posix.split('**')
                if len(pattern_parts) == 2:
                    if name_posix.startswith(pattern_parts[0]) and name_posix.endswith(pattern_parts[1]):
                        return True

    return False

def apply_gitignore_filter(root, dirs, files):
    """Read .gitignore in `root` and return filtered (dirs, files)."""
    gitignore_path = os.path.join(root, '.gitignore')
    patterns = None
    
    print(f"\n  {'-'*50}")
    print(f"  Processing folder: {root}")
    
    # Initialize included.md for this folder
    with open("included.md", "a", encoding="utf-8") as inc_f:
        inc_f.write(f"\n# Folder: {root}\n")
        inc_f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if os.path.exists(gitignore_path):
        patterns = load_gitignore_patterns(gitignore_path)
        print(f"  STEP 1: read .gitignore {gitignore_path}")
    else:
        print(f"  STEP 1: no .gitignore in {root}")

    print(f"  STEP 2: initial dirs={dirs}")
    print(f"  STEP 2: initial files={files}")

    # Filter directories
    kept_dirs = []
    for d in dirs:
        full_d = os.path.join(root, d)
        
        # Check gitignore patterns
        if patterns and gitignore_matches(d, True, patterns):
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_d}\t# excluded: .gitignore pattern\n")
            continue
        
        # Check built-in exclusions
        if should_exclude_folder(full_d):
            continue
        
        kept_dirs.append(d)
        print(f"  STEP 3: keep dir '{d}'")
        with open("included.md", "a", encoding="utf-8") as inc_f:
            inc_f.write(f"{full_d}\t# directory\n")

    # Filter files
    kept_files = []
    for f in files:
        full_f = os.path.join(root, f)
        rel_path = os.path.relpath(full_f, root)
        
        # Check gitignore patterns
        if patterns and gitignore_matches(f, False, patterns):
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_f}\t# excluded: .gitignore pattern\n")
            continue
        
        # Check built-in exclusions
        if f in EXCLUDE_FILES:
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_f}\t# excluded: known excluded filename\n")
            continue
        
        # Check pattern exclusions
        if should_exclude_by_pattern(f):
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_f}\t# excluded: pattern match\n")
            continue
        
        # Special handling for HTML files
        if f.endswith('.html') and not should_keep_html(rel_path):
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_f}\t# excluded: generated HTML report\n")
            continue
        
        # Check if binary
        if is_binary_file(full_f):
            with open("excluded.md", "a", encoding="utf-8") as ef:
                ef.write(f"{full_f}\t# excluded: binary file detected\n")
            continue
        
        kept_files.append(f)
        print(f"  STEP 3: keep file '{f}'")
        with open("included.md", "a", encoding="utf-8") as inc_f:
            inc_f.write(f"{full_f}\t# file\n")

    print(f"  STEP 3: filtered dirs={kept_dirs}")
    print(f"  STEP 3: filtered files={kept_files}")
    print(f"  {'-'*50}\n")
    
    return kept_dirs, kept_files

def has_existing_analysis(folder):
    """Check if folder already has valid analysis files."""
    purpose_file = os.path.join(folder, "auto_gen_purpose.md")
    skills_file = os.path.join(folder, "auto_gen_skills.md")
    
    if FORCE_REANALYZE:
        return False
    
    if os.path.exists(purpose_file) and os.path.exists(skills_file):
        # Check if files are not empty
        if os.path.getsize(purpose_file) > 0 and os.path.getsize(skills_file) > 0:
            return True
    return False

def collect_all_files(folder):
    """Collect all non-excluded files recursively."""
    all_files = []
    
    for root, dirs, files in os.walk(folder):
        # Apply gitignore filter at each level
        gitignore_path = os.path.join(root, '.gitignore')
        patterns = None
        if os.path.exists(gitignore_path):
            patterns = load_gitignore_patterns(gitignore_path)
        
        # Filter directories first (to prevent descending)
        filtered_dirs = []
        for d in dirs:
            if patterns and gitignore_matches(d, True, patterns):
                print(f"  Skipping directory (gitignore): {os.path.join(root, d)}")
                continue
            if should_exclude_folder(os.path.join(root, d)):
                print(f"  Skipping directory (exclude): {os.path.join(root, d)}")
                continue
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs
        
        # Collect files
        for f in files:
            full_f = os.path.join(root, f)
            rel_path = os.path.relpath(full_f, folder)
            
            # Apply exclusions
            if patterns and gitignore_matches(f, False, patterns):
                continue
            if f in EXCLUDE_FILES:
                continue
            if should_exclude_by_pattern(f):
                continue
            
            # Special handling for HTML files
            if f.endswith('.html') and not should_keep_html(rel_path):
                continue
            
            if is_binary_file(full_f):
                # Binary files - silent skip
                continue
            
            # Read file content
            try:
                with open(full_f, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                    if len(content) > 5 * 1024 * 1024:  # 5MB
                        print(f"  ⚠️ Warning: File very large ({len(content)} bytes): {rel_path}")
                        content = f"[FILE VERY LARGE: {len(content)} bytes - showing first 100KB]\n{content[:100000]}"
                    all_files.append((rel_path, content))
                    print(f"  Included: {rel_path} ({len(content)} chars)")
                    with open("included.md", "a", encoding="utf-8") as inc_f:
                        inc_f.write(f"{full_f}\t# analyzed ({len(content)} chars)\n")
            except Exception as e:
                print(f"  ❌ Error reading file {rel_path}: {e}")
                all_files.append((rel_path, f"[ERROR READING: {str(e)}]"))
    
    return all_files

def write_file_list(folder, all_files):
    """Write the list of files to be analyzed to a file for review."""
    file_list_path = os.path.join(folder, "files_to_analyze.txt")
    
    with open(file_list_path, "w", encoding="utf-8") as f:
        f.write(f"# Files to be analyzed in: {folder}\n")
        f.write(f"# Total files: {len(all_files)}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" * 80 + "\n\n")
        
        # Group by file extension for easier review
        by_extension = {}
        for filepath, _ in all_files:
            ext = os.path.splitext(filepath)[1].lower() or "no_extension"
            if ext not in by_extension:
                by_extension[ext] = []
            by_extension[ext].append(filepath)
        
        # Write summary by extension
        f.write("## SUMMARY BY FILE TYPE\n\n")
        for ext in sorted(by_extension.keys()):
            f.write(f"{ext}: {len(by_extension[ext])} files\n")
        
        f.write("\n## DETAILED FILE LIST\n\n")
        for filepath, _ in sorted(all_files):
            # Get file size
            full_path = os.path.join(folder, filepath)
            size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
            size_kb = size / 1024
            f.write(f"{filepath} ({size_kb:.1f} KB)\n")
    
    print(f"\n  📝 File list written to: {file_list_path}")
    return file_list_path

def split_into_chunks(files_with_content, max_context_tokens):
    """Split files into chunks based ONLY on token limit (90% of context)."""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    # Use 90% of context for file contents
    available_tokens = int(max_context_tokens * CONTEXT_SAFETY_FACTOR)
    prompt_overhead = 2000
    available_for_files = available_tokens - prompt_overhead
    
    print(f"\n  Context limit: {max_context_tokens} tokens")
    print(f"  90% available: {available_tokens} tokens")
    print(f"  After overhead: {available_for_files} tokens for files")
    
    for filepath, content in files_with_content:
        file_block = f"\n--- {filepath} ---\n{content}\n"
        file_tokens = count_tokens(file_block)
        
        # Handle oversized files
        if file_tokens > available_for_files:
            print(f"  ⚠️ File too large ({file_tokens} tokens), truncating: {filepath}")
            target_chars = int(available_for_files * 0.8 * 3.5)
            truncated_content = content[:target_chars] + "\n...[FILE TRUNCATED DUE TO SIZE]..."
            file_block = f"\n--- {filepath} ---\n{truncated_content}\n"
            file_tokens = count_tokens(file_block)
            print(f"     Truncated to {file_tokens} tokens")
        
        # Check if adding this file would exceed limit
        if current_tokens + file_tokens > available_for_files and current_chunk:
            chunks.append(current_chunk)
            total_size = sum(c[2] for c in current_chunk)
            print(f"     Chunk {len(chunks)}: {len(current_chunk)} files, {total_size} tokens")
            current_chunk = [(filepath, file_block, file_tokens)]
            current_tokens = file_tokens
        else:
            current_chunk.append((filepath, file_block, file_tokens))
            current_tokens += file_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
        total_size = sum(c[2] for c in current_chunk)
        print(f"     Chunk {len(chunks)}: {len(current_chunk)} files, {total_size} tokens")
    
    return chunks

def analyze_chunk(chunk_files, chunk_index, total_chunks, folder_name):
    """Send a chunk of files to LLM for independent analysis."""
    print(f"\n  📦 Analyzing chunk {chunk_index + 1}/{total_chunks} ({len(chunk_files)} files)...")
    
    chunk_content = []
    for filepath, file_block, _ in chunk_files:
        chunk_content.append(file_block)
    
    system_prompt = f"""You are analyzing a CHUNK ({chunk_index + 1}/{total_chunks}) of files from a folder called "{folder_name}".

Your task: Analyze ONLY the files in this chunk and provide your findings in a structured JSON format.

Based on the files in THIS CHUNK, output a JSON object with:
{{
    "purpose": "What this part of the codebase appears to handle (based on these files only)",
    "skills": ["skill1", "skill2", "skill3"],
    "key_files": ["most_important_file1.ext", "file2.ext"],
    "observations": "Any notable patterns or technologies observed"
}}

Rules:
- Base analysis ONLY on the file contents provided in this chunk
- Be specific about technologies, frameworks, and patterns you actually see
- If no clear purpose is evident from these files, state that honestly

Files in this chunk:
{chr(10).join([f"- {fp}" for fp, _, _ in chunk_files])}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are the files for chunk {chunk_index + 1}:\n\n{''.join(chunk_content)}"}
    ]
    
    try:
        response = send_to_llm(messages, f"chunk_{chunk_index}")
        # Print the raw LLM return for full visibility (user requested actual return)
        print(f"  [LLM RAW RESPONSE - chunk {chunk_index + 1}]\n{response}\n", flush=True)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
        except:
            result = {
                "purpose": response[:1000] if len(response) > 1000 else response,
                "skills": [],
                "key_files": [fp for fp, _, _ in chunk_files[:5]],
                "observations": "Raw response - parsing failed"
            }
        
        return result
    except Exception as e:
        print(f"  ❌ Error analyzing chunk {chunk_index + 1}: {e}")
        return {
            "purpose": f"Error analyzing chunk {chunk_index + 1}: {str(e)}",
            "skills": [],
            "key_files": [],
            "observations": "Analysis failed"
        }

def synthesize_results(chunk_results, folder_name):
    """Combine all chunk analyses into final purpose and skills."""
    print(f"\n  🔄 Synthesizing {len(chunk_results)} chunk analyses...")
    
    summary = f"Folder: {folder_name}\n"
    summary += f"Total chunks analyzed: {len(chunk_results)}\n\n"
    summary += "Individual chunk analyses:\n\n"
    
    for i, result in enumerate(chunk_results):
        summary += f"=== CHUNK {i+1} ===\n"
        summary += f"Purpose: {result.get('purpose', 'N/A')}\n"
        summary += f"Skills: {', '.join(result.get('skills', []))}\n"
        summary += f"Key files: {', '.join(result.get('key_files', []))}\n"
        summary += f"Observations: {result.get('observations', 'N/A')}\n\n"
    
    synthesis_prompt = f"""You are synthesizing multiple chunk analyses into a final assessment for a code folder.

{summary}

Based on ALL the chunk analyses above, produce a FINAL assessment with:

PURPOSE: (2-3 paragraphs describing what this project/folder does overall)

SKILLS: (comma-separated list of ALL technologies, frameworks, languages, and libraries used across the entire codebase)

Rules:
- Combine information from all chunks into a coherent whole
- Do not invent information not present in the chunk analyses
- List each skill only once in the final SKILLS list
- Be specific (e.g., "Python 3.9", "React 18", "FastAPI" rather than just "Python")

Output format:
PURPOSE:
[your purpose statement]

SKILLS:
skill1, skill2, skill3, ...
"""
    
    messages = [
        {"role": "system", "content": "You are a codebase analysis synthesizer."},
        {"role": "user", "content": synthesis_prompt}
    ]
    
    try:
        response = send_to_llm(messages, "synthesis")
        
        purpose_match = re.search(r'PURPOSE:\s*\n?(.*?)(?=SKILLS:|$)', response, re.DOTALL)
        skills_match = re.search(r'SKILLS:\s*\n?(.*?)$', response, re.DOTALL)
        
        purpose = purpose_match.group(1).strip() if purpose_match else "Unable to determine purpose"
        skills = skills_match.group(1).strip() if skills_match else ""
        
        return purpose, skills
    except Exception as e:
        print(f"  ❌ Error synthesizing results: {e}")
        return f"Error during synthesis: {str(e)}", ""

def send_to_llm(messages, round_id=""):
    """Send to LLM and return response."""
    total_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
    print(f"  [LLM] {round_id}: {len(messages)} msgs, {total_tokens} tokens", flush=True)

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }

    max_retries = 3
    pause_seconds = 10

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [LLM] Attempt {attempt}/{max_retries} -> POST {LLM_URL}", flush=True)
            start = time.time()
            resp = requests.post(LLM_URL, json=payload, timeout=3600)

            # If non-2xx, print response body for debugging
            if resp.status_code != 200:
                body_preview = resp.text[:4000]
                print(f"  [LLM ERROR] status={resp.status_code} response_body(<=4k):\n{body_preview}", flush=True)
                resp.raise_for_status()

            # Parse content
            data = resp.json()
            content = data['choices'][0]['message']['content']
            elapsed = time.time() - start
            print(f"  [LLM] {len(content)} chars in {elapsed:.1f}s", flush=True)

            # Remove internal think tags and return
            content = re.sub(r' ?think.*? ?/think ?', '', content, flags=re.DOTALL)
            return content.strip()

        except requests.HTTPError as e:
            last_exc = e
            # If response is attached to exception, print it
            resp = getattr(e, 'response', None)
            if resp is not None:
                try:
                    print(f"  [HTTP ERROR] status={resp.status_code} body:\n{resp.text}", flush=True)
                except Exception:
                    print(f"  [HTTP ERROR] status={resp.status_code} (body unreadable)", flush=True)
            if attempt < max_retries:
                print(f"  [LLM] Retrying in {pause_seconds}s...", flush=True)
                time.sleep(pause_seconds)
                continue
            print(f"  [LLM] Giving up after {attempt} attempts.", flush=True)
            raise

        except requests.RequestException as e:
            last_exc = e
            print(f"  [REQUEST ERROR] {e}", flush=True)
            if attempt < max_retries:
                print(f"  [LLM] Retrying in {pause_seconds}s...", flush=True)
                time.sleep(pause_seconds)
                continue
            print(f"  [LLM] Giving up after {attempt} attempts.", flush=True)
            raise

        except Exception as e:
            last_exc = e
            print(f"  [LLM UNKNOWN ERROR] {e}", flush=True)
            if attempt < max_retries:
                print(f"  [LLM] Retrying in {pause_seconds}s...", flush=True)
                time.sleep(pause_seconds)
                continue
            print(f"  [LLM] Giving up after {attempt} attempts.", flush=True)
            raise

    # If we exit loop unexpectedly, re-raise last exception
    if last_exc:
        raise last_exc
    raise RuntimeError("send_to_llm failed without exception")

def analyze_folder_v2(folder, max_context):
    """Main analysis function - collect all files, chunk based on tokens, analyze, synthesize."""
    print(f"\n{'='*60}")
    print(f"📁 {folder}")
    print(f"{'='*60}")
    
    print(f"\n  📂 Collecting all non-excluded files...")
    all_files = collect_all_files(folder)
    
    if not all_files:
        print("  ❌ No files found after applying exclusions")
        return False
    
    print(f"\n  ✅ Collected {len(all_files)} files")
    
    # Write file list for review (always do this)
    write_file_list(folder, all_files)
    
    # If dry run, stop here (no LLM calls)
    if DRY_RUN:
        print(f"\n  🔍 DRY RUN MODE - No LLM calls made")
        print(f"  📝 Review {os.path.join(folder, 'files_to_analyze.txt')} and adjust exclusions if needed")
        print(f"  🚀 Run without --dry-run to perform actual analysis")
        return True
    
    print(f"\n  ✂️ Splitting into chunks based on token limit...")
    chunks = split_into_chunks(all_files, max_context)
    print(f"  ✅ Created {len(chunks)} chunks")
    
    print(f"\n  🤖 Analyzing chunks independently...")
    chunk_results = []
    for i, chunk in enumerate(chunks):
        result = analyze_chunk(chunk, i, len(chunks), os.path.basename(folder))
        chunk_results.append(result)
    
    print(f"\n  🧠 Synthesizing final results from {len(chunk_results)} chunks...")
    purpose, skills = synthesize_results(chunk_results, os.path.basename(folder))
    
    print(f"\n  💾 Saving results...")
    purpose_path = os.path.join(folder, "auto_gen_purpose.md")
    skills_path = os.path.join(folder, "auto_gen_skills.md")
    
    with open(purpose_path, "w", encoding="utf-8") as f:
        f.write(purpose)
    
    with open(skills_path, "w", encoding="utf-8") as f:
        f.write(skills)
    
    metadata = {
        "folder": folder,
        "total_files": len(all_files),
        "total_chunks": len(chunks),
        "context_limit": max_context,
        "tokens_available_for_files": int(max_context * CONTEXT_SAFETY_FACTOR) - 2000,
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_analyses": chunk_results
    }
    
    metadata_path = os.path.join(folder, "auto_gen_analysis_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✅ Analysis complete!")
    print(f"     - Purpose: {purpose_path}")
    print(f"     - Skills: {skills_path}")
    print(f"     - Metadata: {metadata_path}")
    
    return True

# ========== MAIN ==========
def walk_and_analyze(top_folder, max_context):
    """Walk through folders, analyze each using new chunk-based approach."""
    print(f"\n{'#'*60}")
    print(f"🚀 STARTING: {top_folder}")
    print(f"{'#'*60}")
    if DRY_RUN:
        print(f"🔍 DRY RUN MODE - Only collecting files, no LLM calls")
    print(f"{'#'*60}")
    
    # Initialize log files
    with open("included.md", "w", encoding="utf-8") as inc_f:
        inc_f.write(f"# Agent Folder Analyzer - Included Files Log\n")
        inc_f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        inc_f.write(f"# Top folder: {top_folder}\n")
        inc_f.write("#" * 80 + "\n\n")
    
    with open("excluded.md", "w", encoding="utf-8") as exc_f:
        exc_f.write(f"# Agent Folder Analyzer - Excluded Files Log\n")
        exc_f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        exc_f.write(f"# Top folder: {top_folder}\n")
        exc_f.write("#" * 80 + "\n\n")
    
    analyzed = 0
    skipped_analyzed = 0
    failed = 0
    skipped = 0
    
    for root, dirs, files in os.walk(top_folder):
        filtered_dirs, filtered_files = apply_gitignore_filter(root, dirs, files)
        
        # Track skipped directories
        for skipped_dir in set(dirs) - set(filtered_dirs):
            print(f"  [SKIP] {os.path.join(root, skipped_dir)}")
            skipped += 1
        
        # Prevent descending into excluded directories
        dirs[:] = filtered_dirs
        
        # Analyze if there are files
        if filtered_files:
            # Check if already analyzed (skip in normal mode, but not in dry run)
            if not DRY_RUN and has_existing_analysis(root):
                print(f"\n  ⏭️ SKIPPING (already analyzed): {root}")
                print(f"     Found existing auto_gen_purpose.md and auto_gen_skills.md")
                print(f"     Use --force to re-analyze or delete files to force re-analysis")
                skipped_analyzed += 1
                continue
                
            try:
                success = analyze_folder_v2(root, max_context)
                if success:
                    analyzed += 1
                else:
                    failed += 1
                print(f"\n{'='*60}\n", flush=True)
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted")
                sys.exit(1)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed += 1
        else:
            print(f"\n  ⏭️ Skipping {root} - no files after filtering")
            skipped += 1
    
    print(f"\n{'#'*60}")
    print(f"✅ DONE - Analyzed: {analyzed}, Already analyzed (skipped): {skipped_analyzed}, Skipped (excluded): {skipped}, Failed: {failed}")
    if DRY_RUN:
        print(f"🔍 Dry run complete - No files were analyzed. Review file lists and run without --dry-run when ready.")
    print(f"{'#'*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Top-level folder to analyze")
    parser.add_argument("--context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--model", type=str, default="LOCAL")
    parser.add_argument("--url", type=str, default=LLM_URL)
    parser.add_argument("--force", action="store_true", help="Force re-analysis even if reports exist")
    parser.add_argument("--dry-run", action="store_true", help="Only collect files and write lists, no LLM calls")
    args = parser.parse_args()
    
    MODEL = args.model
    LLM_URL = args.url
    FORCE_REANALYZE = args.force
    DRY_RUN = args.dry_run
    
    if not os.path.isdir(args.folder):
        print(f"❌ Error: Folder '{args.folder}' does not exist")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"⚙️ CONFIG")
    print(f"   Model: {MODEL}")
    print(f"   URL: {LLM_URL}")
    print(f"   Context: {args.context}")
    print(f"   Using 90% of context for files: {int(args.context * CONTEXT_SAFETY_FACTOR)} tokens")
    print(f"   Force re-analyze: {FORCE_REANALYZE}")
    print(f"   Dry run: {DRY_RUN}")
    print(f"{'='*60}")
    
    walk_and_analyze(args.folder, args.context)