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

Status: done

- [x] A 600,000-row CSV whose `code` column holds 0 to 9, with one "N/A" at row 550,000, reads that column as text. `summarize_categorical` reports 11 distinct values and no `mixed_value_types`, and no DtypeWarning appears.
- [x] A 20,000 × 50 file with "x7" at row 18,000 of one column behaves the same way.
- [x] `order_id,city,amount\nA1,Oslo,10,\nA2,Lyon,20,\nA3,Oslo,10,\nA1,Oslo,10,\n` reads as 3 columns with a RangeIndex and the right values. A row with a non-empty extra field raises `unreadable_data_file`, naming its line.
- [x] `score\n1\n\n3\n\n5\n` reads as 5 rows, 2 of them missing. Files with several columns keep skipping blank lines, and the docs say so.
- [x] These columns stay text, and `find_candidate_keys` finds `customer_id` a key:
  - `zip`: "01234", "1234", "02139", "90210";
  - `customer_id`: "007", "7", "0042", "42".
- [x] "-5", "0", "0.5" and "10" stay numbers. Every column without a leading-zero code equals what `pd.read_csv(path, keep_default_na=False, na_values=[""])` gives.
- [x] The header `id,score,score,id` keeps its four labels. `describe_dataset` reports the repeated labels, and another analysis raises its error with the MCP fix: rename the repeated columns in the file.
- [x] `both.tsv` and a symlink `both.csv` pointing to it are each read in their own format within one server.
- [x] The `data_path` description, the MCP instructions and the README state the reading rules: leading zeros, booleans, `inf`, and blank lines. They correct "Parquet keeps its dtypes" for files written without pandas metadata.
- [x] No MCP test's reference read repeats the server's own `read_csv` call.
- [x] The finish checks pass.

## Comments

- Tests are in `tests/test_mcp_reading.py`. Each compares the MCP response with the analysis called on a hand-built DataFrame, or with `pd.read_csv(..., dtype={"zip": str})` for the one test that checks "every other column reads as pandas reads it". The seven other MCP test files had a `read_csv` reference with the server's old call; those references are now hand-built DataFrames. Reads of tables the server writes (`table_path`) still use `pd.read_csv`, since they read output, not the server's input.
- Reading design (`_data_files._read_csv`): pandas reads the file once with `low_memory=False`, `index_col=False` and `skip_blank_lines` off only for a one-column header. A numeric column is read again as text only when a quick byte search finds a field that starts like a leading-zero code, and then only when `read_texts` (the `_parse` rule) confirms one in that column. On 500,000 rows x 40 integer columns the reader takes 1.4 s against 0.5 s for plain `pd.read_csv`; most of the rest is `low_memory=False` and the byte search.
- Extra fields: pandas drops them silently when the first row is wide and rejects them when a later row is wide, and rejects a trailing empty field in that case too. On either signal (the "loss of data" ParserWarning, turned into an error, or the "Expected N fields" ParserError) the file is scanned with the csv module: a non-empty field past the header's raises `unreadable_data_file` naming the line the row ends on, and otherwise the file is read again with names wide enough for the widest row and the extra columns dropped. The message quotes no value, so it can't leak one under masking.
- Repeated header names keep their text; other labels (including `Unnamed: N` for an empty name) are what pandas gives. The `ambiguous_column_labels` error now has an MCP wording: "Rename the repeated columns in the file, then pass its path again."
- The MCP instructions hold 60 lines at most (a contract test), so their reading rules are a shorter text than the `data_path` description, which has the full `CSV_RULES` and `PARQUET_RULES` of `_data_files`. The README states both in full.
- `test_a_file_the_parser_cannot_read_quotes_the_parser` in `tests/test_mcp_data.py` used a ragged TSV as its example of a parser error; that file now gets the new line-naming message, so the example is an unterminated quote.
- Not run: the benchmark (`pytest -m benchmark`), since no analysis changed; the only analysis-side change is the wording of `check_column_labels`' error.
