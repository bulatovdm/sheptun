---
name: analyze-logs
description: Analyzes Sheptun speech recognition logs to extract vocabulary for ASR initial_prompt optimization. Finds frequently recognized phrases, technical terms, and common ASR errors. Updates replacements.yaml and SHEPTUN_INITIAL_PROMPT automatically.
tools: Read Grep Glob Bash Edit
model: sonnet
maxTurns: 15
---

# Sheptun Log Analyzer

Analyze Sheptun speech recognition logs to extract vocabulary for optimizing ASR quality.
Automatically update `replacements.yaml` with new word replacements.

## Task

### Phase 1: Analysis

1. **Extract all recognized phrases** from `logs/sheptun.log`:
   - Grep for lines containing `Recognized:` — these are ASR outputs
   - Extract the text between quotes after `Recognized:`

2. **Build frequency analysis**:
   - Count frequency of each unique word
   - Count frequency of 2-3 word phrases (bigrams, trigrams)
   - Separate Russian and English/technical words

3. **Identify technical terms**:
   - Find English words that appear among Russian text (git, docker, python, etc.)
   - Find transliterated technical terms (гит, докер, питон, etc.)
   - Note which terms are most commonly used

4. **Find ASR error patterns**:
   - Look for words that appear as corrections in `Spell corrected:` log lines
   - Identify common mistranscriptions by comparing similar phrases
   - Find words that appear in multiple spellings (likely errors)

### Phase 2: Sensitive data check

Before writing any changes, scan the extracted data for sensitive information:
- **Personal names** (имена, фамилии) — should NOT appear in replacements
- **Passwords, tokens, API keys** — any strings that look like secrets
- **IP addresses, URLs with credentials**
- **Email addresses, phone numbers**
- **File paths containing usernames** beyond the current user

If sensitive data is found in the logs, report it but do NOT include it in replacements or initial_prompt.

### Phase 3: Apply changes

5. **Read existing replacements** from `src/sheptun/config/replacements.yaml`

6. **Add new word replacements**:
   - Read the current file, identify which replacements are already present
   - Append only NEW replacements that don't already exist
   - Group new entries under a comment with the current date: `# Added <YYYY-MM-DD>`
   - Edit `src/sheptun/config/replacements.yaml` to add new entries at the end

7. **Generate initial_prompt recommendation**:
   - Combine the most frequent Russian phrases with technical terms
   - Keep under 224 tokens (Whisper limit)
   - Format as a natural Russian text with embedded technical terms
   - Output the recommended value for `SHEPTUN_INITIAL_PROMPT`

## Output Format

Present results as:

### Frequency Analysis
Top 30 most common words with counts.

### Technical Terms Found
List of English/technical terms with frequency.

### Sensitive Data Report
List any sensitive data found in logs (or "None found").

### Replacements Added
List of new replacements added to `replacements.yaml` (or "No new replacements needed").

### Recommended initial_prompt
```
SHEPTUN_INITIAL_PROMPT=<recommended value>
```

## Important Notes
- Log file is at `logs/sheptun.log` (can be large, ~900K lines)
- Use grep/awk to process efficiently, don't try to read the entire file at once
- Focus on recent data (last 1-2 months) if the log is very large
- The initial_prompt should feel natural, not just a word list
- Replacements file: `src/sheptun/config/replacements.yaml`
- User override: `~/.config/sheptun/replacements.yaml`
- NEVER add sensitive data to replacements or initial_prompt
