# 04: No Python in record text

What to build: status reasons, limitation reasons, finding claims and field descriptions name arguments and fixes without Python code, so that MCP callers, who read the record, never see Python. The guard test covers output schemas and records.

Read these sections of `.scratch/agent-experience/spec.md`: Records (No Python in record text), and test group 2. This ticket covers user story 11. Findings: STD-39, SPEC-10, AX-27, STD-9, and MCP's questionable decision on Python wording in records.

Blocked by: none

Status: done

- [x] No status reason, limitation reason or finding claim holds code:
  - "make the index unique" replaces `df.reset_index(drop=True)`;
  - "an explicit type conversion" replaces `.astype('float64')`;
  - "the max_pairs limit of 1" replaces `max_pairs=1`;
  - "leaving the columns argument out" replaces `columns=None`.
- [x] Field descriptions hold no `isna()`, `duplicated()`, `df.drop_duplicates(...)` or `scipy.stats.trim_mean(x, 0.1)`, and no mention of `table_path` on the Python side.
- [x] The printed text and error messages keep their surface-specific fixes.
- [x] The guard test from correctness ticket 10 scans every output schema and the records of every recipe.
- [x] The Comments record that this reverses mcp-server ticket 07's choice.
- [x] The finish checks pass.

## Comments

- This reverses mcp-server ticket 07's choice to keep Python wording in records (its Code review follow-up: "Error messages and the record's own text, such as status reasons, still carry Python fixes"). Records now carry no code. Error messages and the printed text's surface-specific lines are unchanged.
- What the ticket settled that the spec left open:
  - The code is gone from record text, not moved: a reason names the limit or the fix in words. A budget reason reads "more than the max_pairs limit of 5,000 allows" and "set numeric_method to pearson"; a row-ID limitation ends "Or make the index unique."; the sentinel limitation no longer lists "(None, NaN, NaT and pd.NA)"; `read as 0 for false and 1 for true`.
  - Of the four examples in box 1, `df.reset_index(drop=True)` and `max_pairs=1` were still in the code; `.astype('float64')` and `columns=None` had already left reasons and claims in earlier tickets (they remained in field descriptions, now reworded).
  - Field descriptions use "the complete table has every column" where they said "(to_frame, or table_path over MCP)": one record schema serves both surfaces, so it names neither. The footer of the printed text and the MCP instructions say how to reach the table. Tables only Python can ask for read "the Python-only table row_distribution".
  - The glossary avoids "null" for a native missing value, so descriptions say "native missing values" and drop the list "None, NaN, NaT and pd.NA". True, False and None, as English words in descriptions and the level order, are written "true", "false" and "the boolean true".
  - Formulas in descriptions and in `definitions` (the standard deviation, the median absolute deviation, skewness and kurtosis, trimmed mean) are words or plain arithmetic without call syntax, and name SciPy's trimmed mean, skewness and excess kurtosis with bias correction.
  - Field paths in `fields` (`columns['x']`) stay: they are paths, not instructions.
- The guard is `tests/test_record_wording.py`, built on `tests/python_syntax.py` from correctness ticket 10. It scans the output schema of every analysis and of the two discovery listings that MCP serves, and the status reason, limitations, findings, `definitions` and top-level text of the record of every analysis on the fixed edge frames. A generated-data run (Hypothesis, with each recipe's options) does the same without the None/True/False check, since a claim may quote a label of the data. Quoted text is data and is skipped. The guard does not need the `mcp` extra, so it runs in the no-extra minimum run too.
- Not scanned: `describe_analysis`'s description records (Python-only: signatures and defaults are Python by design) and error messages (they have their own guard from ticket 10).
- README: the MCP section's sentence on Python syntax now names schemas and record text; the Reference is unchanged. README is 59,861 characters (limit 60,000).
