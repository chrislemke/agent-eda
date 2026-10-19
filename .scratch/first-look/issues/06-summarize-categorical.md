# 06: summarize_categorical

**What to build:** An agent calls `ae.summarize_categorical(df)` and sees, for each categorical column, which values occur and how often, with the denominator stated. It also learns about labels that differ only by case or surrounding spaces, categories that never occur and columns that mix value types, so it can clean them before modelling.

The analysis lives in `agent_eda.statistics.descriptive`. Masking comes in ticket 08, which adds `project` and `mask_sensitive` to this analysis.

Read these sections of `.scratch/first-look/spec.md`: the summarize_categorical entry under The first-look analyses, and Analysis results. Covers user stories 98 to 100.

**Blocked by:** 03-property-tests-and-benchmark, 04-discovery

**Status:** done

- [x] `columns=None` selects object, string, categorical and boolean columns. A numeric column requested by name is summarized by its values.
- [x] Per column, the result gives the non-missing and missing counts, the cardinality, each level with its count and share, the most common levels, the number of levels left out of the display, the number of levels seen once, the unused categories of a categorical dtype, and whether values of different Python types are mixed.
- [x] Shares use the non-missing count as their denominator, and the result states it.
- [x] Levels are ordered by count, descending, then by label. Labels of mixed types follow a documented order that never depends on hash order. Labels are kept exactly as stored.
- [x] `top_n`, default 10, limits only the display, and the result records it. The record caps levels at 100 with the total, and `to_frame()` returns every level.
- [x] Labels that differ only by case or surrounding spaces, such as "Yes", "yes " and "YES", give a suspected finding. Unused categories give a descriptive finding and mixed value types a suspected one.
- [x] The declared statuses are `completed`, `partial`, `insufficient_data`, `not_applicable` and `failed`, and they follow the per-column rules from ticket 02.
- [x] Counts and shares match pandas `value_counts` for object, pandas 3 string, categorical, boolean and pyarrow-backed columns.
- [x] summarize_numeric's `ColumnTypeError` for a text or categorical column now names summarize_categorical as the analysis that fits.
- [x] The docstring has the seven sections in order and passes the contract tests.
- [x] The property suite has a call recipe for it, and the properties hold.
- [x] With default arguments it finishes in under 5 seconds on the benchmark DataFrame.
- [x] `ae.search_analyses("value counts")` returns it first. The row joins the search acceptance table, and every earlier row still passes.
- [x] Its catalogue entry is marked implemented, with its differences from the catalogue noted.
- [x] Nearby analyses mention it in `When to use:` and `See Also:` where it is the better choice.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **A level is what pandas counts as one.** Values form one level exactly when `pd.factorize` and `value_counts` treat them as equal. So 1, 1.0 and True are one level, which keeps the label seen first, and -0.0 joins 0.0. Missing means factorize's code -1, which covers None, NaN, NaT, pd.NA and Decimal NaN and gives the same count as describe_dataset. NaN that a pyarrow-backed float column stores as a value is a level, as in `value_counts`. Unused categories are no levels. `value_counts` lists them with a count of 0, while the entry lists them in `unused_categories`, in category order, so the value_counts test drops the zero rows.
- **Fields.** Each entry has `non_missing_count`, `missing_count`, `cardinality`, `levels`, `most_common_levels`, `left_out_level_count`, `seen_once_count`, `unused_categories`, `value_types`, `mixed_value_types` and `case_or_space_variants`. A level has `label`, `label_type`, `count` and `share`. The result adds `left_out_columns`; `share_denominator`, which names the field every share divides by; and `level_order`, which states the order in words. `most_common_levels` holds the first top_n levels, and `left_out_level_count` counts the rest. `top_n` runs from 1 to 100, since the entry and the record keep 100 levels.
- **The entry keeps 100 levels too.** `levels` holds the first 100 levels on the result object as well as in the record, and `to_frame()` has every level. The record's `levels_total` is the cardinality. For this, `_Cap` in `agent_eda.results` gained `total=`, which names the field that holds a list's total when the object keeps only the first entries. Every level as a pydantic object would miss the 5-second target, since the benchmark frame has 5,397,437 levels. The complete table lives in the private `_tables` attribute, one `Levels` per column, and `to_frame()` sorts it when called. `model_copy`, `copy.deepcopy` and pickling keep it, checked on pydantic 2.7.0 and 2.14.1. The first 100 levels need no full sort. A partition finds the count of the 100th level, and only labels with that count get compared, text labels with one vectorized comparison.
- **The table keeps labels as pandas holds them.** `Levels.values` is the index of distinct values that `pd.factorize` or the categories give, and steps that need Python objects convert them on the spot. On pandas 3, pyarrow-backed text would otherwise become 5.4 million Python strings that the result keeps alive. Holding Python strings, the call raised the peak memory 0.41 GiB above the frame's; now it raises it 0.23 GiB, and on pandas 2.2, whose object columns already hold Python strings, it adds nothing.
- **Order.** Count descending, then label. Booleans and numbers sort first, by value, with NaN last. Text follows by code point, then other types by type name and string form. Labels that still tie keep the data's order, which is category order for a categorical. Ordered categoricals follow the same rule. A test runs the same call in two processes with different `PYTHONHASHSEED` values and compares the JSON.
- **Labels.** The record holds a str, bool, int or finite float label as itself and any other label as its string form. `label_type` names the Python type, and a NumPy scalar counts as the Python type it holds. `to_frame()` keeps the stored values, so `df[col].isin(frame["label"])` works. The printed text quotes text labels, which keeps the space in 'yes ' visible.
- **Case and space variants.** Text labels match after `str.strip()` and `str.casefold()`, so tabs and other surrounding whitespace count, and 'Straße' matches 'STRASSE'. When no label changes under strip and casefold, the check stops after one pass. That holds for every text column of the benchmark. The claim says "surrounding spaces", the ticket's words, and the field description says what counts. Claims name columns and never labels, so ticket 08's masking has to cover the fields, the text table and `to_frame()`, but not the claims.
- **Mixed value types.** `value_types` lists the type names of the non-missing values. `value_counts` merges 1, 1.0 and True, so a column of objects whose labels aren't all text gets checked value by value. The code lives in `value_types` in `agent_eda/_levels.py`. infer_column_semantics, ticket 12, has a `mixed_value_types` flag too and should call it, so the two can't disagree.
- **Statuses.** A column without a non-missing value is `insufficient_data`, code `no_values`. A column of values that can't be hashed, such as lists or dicts, is `not_applicable`, code `unhashable_values`. Its counts of levels are null and its level lists empty, the existing `unhashable_values` limitation names the null fields, and its `value_types` still lists the types. Data without a text, object, categorical or boolean column gives `not_applicable` with code `no_categorical_columns`. `failed` is declared, as the ticket asks, but no code path returns it.
- **Columns.** `columns=None` selects text, object, categorical and boolean columns. A numeric column named in `columns` gets counted by its values. Datetime, duration and other columns, such as pyarrow lists, raise `ColumnTypeError` with the fix `df[col].astype(str)`, since the spec accepts only numeric columns by name.
- **Errors name the analysis that fits.** `ColumnRule` gained `alternatives`, which maps a rejected kind to the analysis that fits it. For a text, object or categorical column, summarize_numeric's message now starts its fixes with "Use summarize_categorical for 'city' instead".
- **Text.** The table has one row per column, with `non_missing_count`, `missing_count`, `cardinality` and the most common levels written as `'basic': 3 (60%)`. A long cell gets cut at 120 characters. `_text.Table` gained `frame_rows` and `frame_unit`, so the footer counts what `to_frame()` holds, which here is levels.
- **Shared code.** `agent_eda/_levels.py` holds the counting, the order, the record form of labels and the levels table, for profile_target's class counts in ticket 15. `results.py` imports it, so it must not import any module that imports `results.py`.
- **Search tags.** "value counts", "frequencies", "levels", "categories", "cardinality", "most common values", "unused categories" and "case variants". "value counts" scores 6 against summarize_numeric's 3. Without tags the two tied at 3, and the alphabetical tie-break decided.
- **Glossary gap.** The spec and this ticket use "level" throughout, but GLOSSARY.md has no entry for it. I left that for /domain-modeling.

Measurements, two runs each, on an Apple M3 Max with 16 cores under macOS 27.0.1, Python 3.12.11 and 2 threads, with `uv run pytest -m benchmark` and the minimum-versions command. Other agents shared the machine, so the describe_dataset and summarize_numeric columns show how close each run came to ticket 03's table:

| Versions | summarize_categorical | describe_dataset | summarize_numeric | Peak memory |
| --- | --- | --- | --- | --- |
| pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0 | 1.17 and 1.17 s | 0.95 and 0.81 s | 1.29 and 1.27 s | 1.94 GiB |
| pandas 2.2.0, numpy 1.26.0, pyarrow 14.0.1 | 1.47 and 1.58 s | 2.13 and 2.02 s | 1.43 and 1.46 s | 2.25 GiB |

- Factorizing the text columns is most of the time, about 1 s of it on pandas 2.2.0. The case and space check costs about 60 ms per million distinct labels. Choosing the first 100 levels of a million distinct text labels costs about 30 ms, where a heap with a key function took 58 to 84 ms.
- An earlier run under heavier load took 2.49 s on pandas 2.2.0, while describe_dataset took 2.90 s. The slowest case so far, describe_dataset on pandas 2.x, stays the one to watch on the 2-vCPU target.

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass. So does `uv run --isolated --resolution lowest-direct --python 3.12 pytest`, with numpy 1.26.0, pandas 2.2.0, pydantic 2.7.0 and pyarrow 14.0.1, and `uv run --isolated --python 3.1x pytest` on Python 3.13 and 3.14.
- The `thorough` profile passed. Random fuzzing passed too, with seeds 1 to 6, 1,500 examples each and frames of up to 30 rows and 8 columns, half on each version set. It checked the contract properties and a new property: counts match `value_counts`, and the entry's levels, the most common levels and `to_frame()` agree.
- Linux x86_64 and ARM64 didn't run locally; ticket 05's CI covers them. The target machine, a 2-vCPU AgentCore session, is still unmeasured.

Box 14 is superseded: commit b430a0d deleted the catalogue (`docs/eda-python-implementation-catalogue.md`) and moved the planned analyses into the README, so no catalogue entry is marked implemented any more. The code and docstring of a shipped analysis are authoritative.
