---
name: analyze-logs
description: Analyzes Sheptun speech recognition logs to extract vocabulary for ASR initial_prompt optimization. Finds frequently recognized phrases, technical terms, and common ASR errors. Updates replacements.yaml and SHEPTUN_INITIAL_PROMPT automatically.
tools: Read Grep Glob Bash Edit
model: sonnet
maxTurns: 25
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

## Efficient Log Processing

The log file (`logs/sheptun.log`) can be very large (~900K lines). You MUST use these one-liner shell commands — do NOT try to read the file directly or use Python.

**Extract all recognized phrases:**
```bash
awk -F"Recognized: '" '/Recognized:/{sub(/.$/,"",$2); print $2}' logs/sheptun.log > /tmp/sheptun_phrases.txt
```

**Top Russian words by frequency:**
```bash
cat /tmp/sheptun_phrases.txt | tr ' ' '\n' | grep -E '^[а-яёА-ЯЁ]+$' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn | head -50
```

**Top English/technical terms:**
```bash
cat /tmp/sheptun_phrases.txt | tr ' ' '\n' | grep -iE '^[a-z][a-z0-9._-]+$' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn | head -40
```

**Find transliterated tech terms (potential replacements):**
```bash
cat /tmp/sheptun_phrases.txt | tr ' ' '\n' | grep -E '^[а-яёА-ЯЁ]+$' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn | grep -iE 'комит|пуш|мердж|мёрдж|бранч|бренч|деплой|диплой|ребейз|фетч|стейдж|докер|кубер|нжинкс|пайтон|питон|тайпскрипт|реакт|бэш|баш|гит|фигм|слак|телеграм|ютуб|клауд|виспер|плейрайт|джейсон|ямл|ридми|вскод|ларавел|пхп|постгрес|битбакет|чекаут|продакшен|продакшн'
```

IMPORTANT: macOS grep does NOT support `-P` flag. Use `-E` for extended regex.

## Important Notes
- NEVER read the log file with the Read tool — it's too large
- Run shell commands first, analyze output, then edit replacements.yaml
- Budget your turns: analysis in 5-6 turns, editing in 2-3 turns, report in 1-2 turns
- Replacements file: `src/sheptun/config/replacements.yaml`
- User override: `~/.config/sheptun/replacements.yaml`
- NEVER add sensitive data to replacements or initial_prompt
