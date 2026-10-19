# 07: MCP CSV reading keeps values as written

What to build: the MCP server reads a CSV so that every analysis sees the values the file holds:
- one type per column whatever the file's length;
- no data column used as the index;
- empty lines of a one-column file kept as missing values;
- codes with leading zeros kept as text;
- repeated header names kept;
- a cache that knows the file format.

Read these sections of `.scratch/correctness/spec.md`: Reading CSV files over MCP, and test group 6. This ticket covers user story 20. Findings: STD-1, MCP-1, MCP-2, STD-3, SPEC-2, MCP-12, SPEC-6, MCP-14, STD-11.

Blocked by: none

Status: ready-for-agent

- [ ] A 600,000-row CSV whose `code` column holds 0 to 9, with one "N/A" at row 550,000, reads that column as text. `summarize_categorical` reports 11 distinct values and no `mixed_value_types`, and no DtypeWarning appears.
- [ ] A 20,000 × 50 file with "x7" at row 18,000 of one column behaves the same way.
- [ ] `order_id,city,amount\nA1,Oslo,10,\nA2,Lyon,20,\nA3,Oslo,10,\nA1,Oslo,10,\n` reads as 3 columns with a RangeIndex and the right values. A row with a non-empty extra field raises `unreadable_data_file`, naming its line.
- [ ] `score\n1\n\n3\n\n5\n` reads as 5 rows, 2 of them missing. Files with several columns keep skipping blank lines, and the docs say so.
- [ ] These columns stay text, and `find_candidate_keys` finds `customer_id` a key:
  - `zip`: "01234", "1234", "02139", "90210";
  - `customer_id`: "007", "7", "0042", "42".
- [ ] "-5", "0", "0.5" and "10" stay numbers. Every column without a leading-zero code equals what `pd.read_csv(path, keep_default_na=False, na_values=[""])` gives.
- [ ] The header `id,score,score,id` keeps its four labels. `describe_dataset` reports the repeated labels, and another analysis raises its error with the MCP fix: rename the repeated columns in the file.
- [ ] `both.tsv` and a symlink `both.csv` pointing to it are each read in their own format within one server.
- [ ] The `data_path` description, the MCP instructions and the README state the reading rules: leading zeros, booleans, `inf`, and blank lines. They correct "Parquet keeps its dtypes" for files written without pandas metadata.
- [ ] No MCP test's reference read repeats the server's own `read_csv` call.
- [ ] The finish checks pass.

## Comments
