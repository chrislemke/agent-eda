# 17: README and 0.1 acceptance check

**What to build:** Someone who lands on the repo sees what 0.1 does and how to use it, in examples that run as written. A final pass then checks the slice against the spec as a whole, now that all 11 analyses, discovery, the Project with masking, packaging and CI are in place.

Read these sections of `.scratch/first-look/spec.md`: Docs and records, plus the search acceptance cases and test group 9 under Testing Decisions. Covers user stories 115 and 123.

**Blocked by:** 05-ci-workflow, 09-project-files, 11-detect-missing-sentinels, 13-validate-column-types, 14-find-candidate-keys, 15-profile-target, 16-compute-association-matrix

**Status:** ready-for-agent

- [x] The README's planned usage becomes real usage that runs as written, and its status line says what 0.1 contains.
- [x] The README shows the first-look order and how to switch on masking, and calls masking a guardrail, not anonymization or access control.
- [x] A test runs the README's usage examples.
- [x] Every row of the spec's search acceptance table passes.
- [x] `ae.guide()` names all 11 analyses in the first-look order and stays within about 60 lines.
- [x] All 11 analyses finish in under 5 seconds on the benchmark with default arguments. The timings and the machine go under `## Comments` in this ticket.
- [x] All 11 catalogue entries are marked implemented. Between them they note every difference the spec lists: no `context` argument, literal methods, defaults in compute_association_matrix instead of required type and method mappings, `max_key_size` defaulting to 2, weights deferred, "convertible" instead of "parseable", and shares instead of rates.
- [x] `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, on the latest and on the minimum dependency versions.

## Comments

What this ticket settled that the spec left open:

- **The first-look order names all 11 analyses.** The spec's order names 10 and gives validate_column_types no step, while the box asks the guide to name all 11 in the first-look order. So validate_column_types joins step 2 next to infer_column_semantics, which the guide describes as "logical types, inferred and checked against your schema". It needs a declared schema, just as profile_target in step 7 needs a target column and a task. `discovery.FIRST_LOOK` holds the order. `test_names_all_eleven_analyses_in_the_first_look_order` checks that the guide prints exactly the test's own copy, no longer filtered by what is registered. Ticket 04's comment already described FIRST_LOOK as holding all 11 names. It held 10 until now.
- **Project files in the guide.** Ticket 09 left them to this ticket. The masking paragraph gained one line: "Save it once with project.save("project.toml") and load it in each later script with ae.Project.load("project.toml")." The guide now prints 60 lines, the test's cap, of at most 117 characters. Its listing takes a line per analysis, so the next slice that adds analyses has to shorten the page.
- **README.** "Planned usage" became "Usage". The example builds an 8-row customer DataFrame, so every block runs as written. The data holds what the spec's problem statement warns about: a -999 that pulls the mean age down to -86.12, 'Basic ' next to 'basic', and one customer recorded twice under the IDs 102 and 108. The section then shows:
  - summarize_numeric's call and printed text, then `to_frame()`, `to_json()` and a typed field;
  - the three discovery calls;
  - the first-look order as a table of step, analysis and the question it answers, followed by a loop that runs all 11 in that order;
  - masking with a DataPolicy and its printed text, how the per-call setting wins, that numeric statistics and index labels are never masked, and saving and loading the project;
  - "Masking is a guardrail against accidental exposure, not anonymization or access control.", the docstrings' sentence, with ADR 0001's reason.

  The status line says what 0.1 contains. The sentence about dataset roles is gone, since 0.1 has none. Installation says how to build the wheel until 0.1 is on PyPI, and the Development section says what `tests/test_readme.py` checks.
- **Shown output.** The README shows only output that both version sets print alike, which holds for summarize_numeric and the masked summarize_categorical. describe_dataset, for one, prints the text dtype as str on pandas 3 and object on pandas 2.
- **`tests/test_readme.py`** runs the README's python blocks in order in one namespace, as if pasted into one session. It runs them from an empty temporary directory, because one block saves a project file. When the next fenced block after a python block is a `text` block, it must equal what the python block prints. Changing one number in the shown output failed the test. A second test reads the first-look steps from `ae.guide()` and requires the README's table to list them in order, each with the analysis's summary from `ae.list_analyses()`. A renamed analysis or an edited summary line fails it. A third test requires the guardrail sentence.
- **`tests/test_catalogue.py`** requires each analysis in `ae.list_analyses()` to have exactly one catalogue row marked "**Implemented in agent-eda X.Y** as `module.name`" with an "**Implemented differently:**" list, which AGENTS.md asks for when an analysis ships. Naming the wrong module in find_candidate_keys' row failed it.
- **Catalogue.** The analysis tickets had marked all 11 entries, and none needed a change. A scratch script found where they note each difference the spec lists:
  - no `context` argument: all 11;
  - literal methods: compute_association_matrix, and profile_target's `task`;
  - defaults instead of the required `column_types` and `methods` mappings: compute_association_matrix;
  - `max_key_size` defaulting to 2: find_candidate_keys;
  - weights deferred: summarize_numeric, summarize_categorical and profile_target;
  - `convertible_count` instead of parseable_count: validate_column_types;
  - shares instead of rates: infer_column_semantics, profile_missing_values, detect_missing_sentinels and find_duplicate_rows.
- **Search.** All 12 rows of the spec's table pass in `test_an_agents_wording_finds_the_right_analysis_first`, which already held them. The narrowest lead is "value counts", where summarize_categorical scores 6 and profile_missing_values 5. Every other expected analysis leads by at least 2.
- **The final pass against the spec.** On top of the suite, a scratch script checked through the public API and found no difference from the spec. It covered the top-level exports and, for each analysis, its module, its declared statuses, its masking support, its result class name and `__test__`. The statuses are the spec's plus `failed`, and exactly the five analyses the spec names support masking. `data` is the one positional parameter, and every option is keyword-only under the spec's name and default. The script also checked the built-in bases of the five error classes, and that `import agent_eda` and all four discovery functions leave SciPy, statsmodels and scikit-learn unimported in a fresh interpreter.

Measurements, from `uv run pytest -m benchmark` and the minimum-versions command with `-m benchmark`, two runs each. The machine is an Apple M3 Max with 16 cores, 12 of them performance cores, and 128 GiB under macOS 27.0.1, with Python 3.12.11. The benchmark limits BLAS and pyarrow to 2 threads. No other agent was running, and the load average stayed near 3 of 16 cores, the benchmark included.

| Analysis | pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0 | pandas 2.2.0, numpy 1.26.0, pyarrow 14.0.1 |
| --- | --- | --- |
| describe_dataset | 0.77 and 0.76 s | 1.72 and 1.74 s |
| infer_column_semantics | 1.48 and 1.48 s | 1.88 and 1.90 s |
| validate_column_types | 1.34 and 1.35 s | 1.63 and 1.65 s |
| profile_missing_values | 0.08 and 0.08 s | 0.32 and 0.33 s |
| detect_missing_sentinels | 0.58 and 0.59 s | 1.38 and 1.39 s |
| summarize_numeric | 1.16 and 1.19 s | 1.34 and 1.38 s |
| summarize_categorical | 1.06 and 1.05 s | 1.31 and 1.31 s |
| find_duplicate_rows | 0.04 and 0.04 s | 0.04 and 0.04 s |
| find_candidate_keys | 0.36 and 0.36 s | 0.50 and 0.51 s |
| profile_target | 0.01 and 0.01 s | 0.01 and 0.01 s |
| compute_association_matrix | 0.84 and 0.84 s | 1.38 and 1.39 s |
| Frame built in | 2.8 and 2.8 s | 2.4 and 2.3 s |
| Peak memory | 2.08 and 2.05 GiB | 2.24 and 2.24 GiB |

- Every analysis completed. The slowest is infer_column_semantics on the minimum versions at 1.90 s, 2.6 times below the 5-second target on this machine. Ticket 12 measured it at 2.29 and 3.41 s while other agents slowed the machine about 1.5 times.
- Calls beyond the defaults are slower, as their tickets measured. The spec's target covers default arguments on the benchmark frame, so none of these fails it:
  - masking all 35 text and categorical columns of the benchmark: summarize_categorical takes 5.2 s on the latest and 6.2 s on the minimum versions (ticket 08);
  - a declared `datetime_format`: strptime reads about a million distinct matching values in 4 s (ticket 13);
  - frames of 90 categorical columns with 12 to 24 levels: compute_association_matrix takes up to 10 s (ticket 16).

Checks:

- `uv run ruff check .`, `uv run ruff format .`, `uv run ty check` and `uv run pytest` pass, with 1,828 tests passed and 2 skipped. `uv run --isolated --resolution lowest-direct --python 3.12 pytest` passes the same 1,828 with numpy 1.26.0, pandas 2.2.0, pydantic 2.7.0, scipy 1.11.2 and pyarrow 14.0.1, and leaves `uv.lock` unchanged. `ruff check .` and `ruff format --check .` pass in that environment too. `uv run --isolated --python 3.13 pytest` and the same on 3.14 pass.
- With ty 0.0.89 and pandas 2.2.0, `ty check` in the minimum-versions environment reports 30 diagnostics. First-look's tip, ca29ae4, has the same 30, and none is in this ticket's files. They come from pandas 2.2.0's own annotations: `NaTType` has no `isoformat`, `StringDtype` takes no `na_value`, and lists don't satisfy its `SequenceNotStr` protocol. Neither the README nor CI runs ty on the minimum versions, and the minimum-versions test run covers that code, so I left them for a decision of their own.
- `ruff format .` formats the README's python blocks too. They were already formatted.

Not checked locally, or still open:

- Linux x86_64 and ARM64, which ticket 05's CI covers once the repo is on GitHub. The 2-vCPU, 8 GB target, whose per-core speed against an M3 Max core nobody has measured. The benchmark's peak memory of 2.24 GiB fits in 8 GB.
- Two open points from earlier tickets, outside this one. pandas' factorize stops text at a NUL character, so summarize_categorical, describe_dataset and find_duplicate_rows still count '' and '\x00a' as one value (ticket 11). And the glossary gaps that tickets 06, 10, 13, 14 and 15 left for /domain-modeling remain: level, native missing value, compatible and convertible values, key and primary key, and target, class and label.
