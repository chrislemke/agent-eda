# Spec: agent-eda foundation and first-look analyses

Status: done

## Problem Statement

An agent that starts exploring a new dataset today writes its own pandas code: `df.describe()`, `df.isna().sum()`, `df.duplicated()`, `df.corr()`. That serves an agent badly.

The output is wide, untyped text. `describe()` over 100 columns costs thousands of tokens. AgentCore Code Interpreter hands back only what was printed, Claude Code cuts MCP tool output at 25,000 tokens, and Strands trims old tool results when the context fills up. The agent pays for numbers it never reads and can lose the ones it needed.

Ad-hoc code hides statistical mistakes. Sentinel values such as -999 or "N/A" get averaged as real data. Postal codes become numbers. A numeric class label looks like a regression target. `df.corr()` quietly drops categorical columns and uses one measure nobody chose. A constant column yields NaN that the agent reads as a value. Nothing tells the agent which definition or denominator it got.

Failures look different every time. An empty column might raise, return NaN or print nothing, so the agent can't tell "too little data" from "doesn't apply" from "my call was wrong". A misspelled column name raises a bare KeyError with no hint.

The agent can't see what's worth checking. It works from memory and skips whole areas, such as candidate keys, stand-in values for missing data, or labels that differ only by case.

The environments add limits of their own. An AgentCore Code Interpreter sandbox can't reach PyPI and only has its preinstalled libraries. People doing EDA by hand hit the same problems on a smaller scale.

## Solution

`agent-eda` 0.1 is a pure-Python library, imported as `agent_eda`, that gives an agent or a person a correct and compact first look at any pandas DataFrame.

- Eleven first-look analyses cover the dataset's structure, each column's logical type, declared types, missing values, sentinel values, numeric and categorical summaries, duplicate rows, candidate keys, the target, and associations between columns.
- Each analysis is a plain, deterministic function. The DataFrame comes first, everything else is a typed keyword argument, and nothing has to be set up.
- Each analysis returns a typed analysis result. Printing it shows a short summary of fixed maximum size: status, key numbers, findings, limitations and where to find more. `to_frame()` returns the complete tables, and `to_dict()` and `to_json()` return a strict-JSON record.
- Eight statuses say how a run ended. Findings with assessments say what the data showed. A mistake in a call raises an exception that explains the fix.
- `ae.guide()`, `ae.search_analyses()`, `ae.list_analyses()` and `ae.describe_analysis()` let an agent find and understand analyses without reading source code.
- An optional Project holds the data policy's sensitive columns, so an agent can switch on masking before raw values reach its model provider.
- The wheel installs with pip, installs offline with `--no-deps`, or imports straight from the file on `sys.path`. It works the same in AgentCore Runtime, AgentCore Code Interpreter, coding agents and notebooks.

## User Stories

### Installing and running anywhere

1. As a data scientist, I want to install agent-eda with pip or uv and import it as `agent_eda`, so that I can use it like any other package without setup.
2. As an AgentCore agent in a Code Interpreter session with public network access, I want `pip install agent-eda` to succeed with the preinstalled numpy, pandas, scipy, statsmodels, scikit-learn and pydantic, so that installing doesn't replace libraries the sandbox already has.
3. As an AgentCore agent in a sandbox session without PyPI access, I want to install an uploaded wheel with `pip install --no-deps`, so that I can use the package offline.
4. As an AgentCore agent in a sandbox that blocks pip, I want to import the package straight from the uploaded wheel file on `sys.path`, so that every analysis still works.
5. As an agent developer deploying to AgentCore Runtime, I want a pure-Python wheel with only six required dependencies, so that it installs on ARM64 and stays well within the runtime's size limits.
6. As an agent in an environment I can't upgrade, I want every analysis to work with numpy 1.26 or 2.x and pandas 2.2 or 3.x, so that I can use the libraries already installed.
7. As an agent, I want `import agent_eda` to load without importing SciPy, statsmodels or scikit-learn, so that my first call is fast.
8. As a data scientist, I want `agent_eda.__version__` to report the installed version, so that I can cite it when I share results.

### Finding the right analysis

9. As an agent new to the package, I want `ae.guide()` to print a one-page overview of what the package does, how to call an analysis, how to read results and which analyses exist, so that I can start without reading source code.
10. As an agent starting EDA on a new dataset, I want the guide to suggest an order for a first look, so that I cover structure, types, missing values, duplicates, distributions, the target and associations without forgetting one.
11. As an agent, I want `ae.search_analyses()` with a plain-language question to return the best-matching analyses with one-line summaries, so that I find the right analysis without knowing its name.
12. As an agent, I want search to understand common synonyms such as nulls or NaN for missing values, dupes for duplicates, correlation for association and label for target, so that my wording doesn't have to match the package's.
13. As an agent, I want the same search query to return the same ranked list every time, so that my workflow is reproducible.
14. As an agent, I want a search without matches to say so and point me to the guide, so that I don't get a silent empty answer.
15. As an agent, I want `ae.list_analyses()` to list every analysis with its module and one-line summary, so that I can see the whole set at once.
16. As an agent, I want `ae.describe_analysis()` to show an analysis's parameters with types, defaults, allowed values and meanings, its statuses, finding codes, result fields, masking support and an example call, so that my first call is correct.
17. As an agent, I want a misspelled name in `describe_analysis()` to raise an error that suggests the closest analysis names, so that I fix it in one step.
18. As an agent, I want discovery functions to run without loading SciPy or other heavy libraries, so that exploring the package is cheap.
19. As an agent, I want each docstring's first line to be the question the analysis answers, so that `help()` tells me at a glance whether it fits.
20. As an agent, I want each docstring to say when to use the analysis and which analysis to use instead for nearby questions, so that I pick the right one.
21. As an agent, I want each docstring to name the result fields to read and explain how to interpret them, so that I draw correct conclusions.
22. As an agent, I want a short runnable example in each docstring, so that I can copy a working call.
23. As an agent developer, I want Google-style docstrings with an `Args:` section, so that Strands, LangChain, the OpenAI Agents SDK and Pydantic AI read parameter descriptions correctly if I wrap analyses as tools later.
24. As a data scientist, I want editor completion and type hints for analyses, parameters and result fields, so that I can explore the API from my IDE.

### Calling an analysis

25. As an agent, I want every analysis to take the DataFrame first and all other arguments by keyword, so that I can't pass arguments in the wrong order.
26. As an agent, I want to run any analysis on a plain DataFrame without a Project or any configuration, so that my first call just works.
27. As an agent, I want `columns=None` to mean every suitable column for that analysis, so that a first call needs no column list.
28. As an agent, I want parameters that take column names to end in `column` or `columns`, so that I can tell which arguments expect names.
29. As an agent, I want fixed choices such as methods and task types typed as literals, so that I see the allowed values in the signature and a typo fails immediately.
30. As an agent, I want standard conventions such as quantile levels and the trimming fraction to have documented defaults that the result records, so that I know what was used without passing everything.
31. As an agent, I want analyses never to invent project-specific values such as a rare-class share or the expected labels, so that conclusions rest only on what I declared.
32. As an agent, I want analyses to leave my DataFrame untouched, including values, dtypes, index and column order, so that I can run them in any order.
33. As an agent, I want analyses to accept nullable pandas dtypes, categoricals, pandas 3 strings and pyarrow-backed columns, so that I don't have to convert data loaded from Parquet or CSV.
34. As a data scientist, I want calls that look like ordinary pandas or SciPy calls, so that there's nothing new to learn.

### Reading results

35. As an agent, I want `print(res)` to show a short plain-text summary with the status, key numbers, findings, limitations and how to get more, so that results don't flood my context window.
36. As an agent, I want the printed summary to stay within a fixed size however large the data, and to say what it left out (for example "10 of 40 columns shown"), so that one call can't blow up my context.
37. As an AgentCore agent, where only printed output reaches me, I want the printed summary to be enough to decide my next step, so that I rarely need a second call.
38. As an agent, I want `res.to_frame()` to return the complete table behind a result as a DataFrame, so that I can filter, sort or join it in pandas.
39. As an agent, I want results with several tables to let me choose one by name in `to_frame()`, and the printed summary to name them, so that I can reach every table.
40. As an agent, I want `res.to_dict()` and `res.to_json()` to give the whole result record as strict JSON, so that I can store it or hand it on.
41. As an agent, I want undefined values to appear as null with a stated reason instead of NaN or infinity, so that JSON parsers don't break and I know why a value is missing.
42. As an agent, I want typed attributes on every result, such as the median of one column, so that I can use values in code without parsing text.
43. As an agent, I want every result to state the analysis, the method, the parameters actually used (defaults and the resolved column list included) and the library versions, so that I can explain and rerun what happened.
44. As an agent, I want two identical calls in the same environment to give byte-identical JSON, even in different processes, so that I can compare runs and trust reruns.
45. As an agent, I want every result to carry its support (rows given, values eligible, values evaluated, values excluded and why), so that I know every denominator.
46. As an agent, I want findings with an assessment (descriptive, suspected, violation, passed or unknown), so that I can tell neutral facts from possible problems and broken rules.
47. As an agent, I want each finding to carry a stable code and point at the result fields behind it, so that I can branch on codes and verify claims.
48. As an agent, I want limitations listed with stable codes and the fields they affect, so that I don't over-read a number.
49. As an agent, I want lists that grow with the data (row IDs, example values, levels, groups) capped in the result record together with their totals, so that the record stays small and I still know the true counts.
50. As an agent, I want rows identified by their DataFrame index labels, so that `df.loc[ids]` shows me the rows right away.
51. As an agent, I want to pass `id_column=` to identify rows by a column instead, so that row IDs mean something when my index is a plain row number.
52. As an agent, I want a limitation when row IDs are ambiguous because the index or ID column has duplicates, so that I don't look up the wrong rows.
53. As a data scientist in Jupyter, I want results to display as the same readable text, so that I see what an agent sees.

### Statuses and errors

54. As an agent, I want every result to carry one of eight statuses, so that I can branch on how the run ended.
55. As an agent, I want `insufficient_data` when there is too little data, such as an empty DataFrame or a column without values, so that I know to get more data or pick another analysis.
56. As an agent, I want `not_applicable` when the data has nothing the analysis can work on, such as no numeric columns for a numeric summary, so that I move on.
57. As an agent, I want `partial` when some columns completed and others didn't, with each incomplete column and its reason listed, so that I can use what did complete.
58. As an agent, I want `budget_exceeded` when an analysis would exceed its work limit, with the estimate and the limit, so that I can raise the limit or narrow the columns.
59. As an agent, I want `failed` with an error code and message when a known numerical failure occurs, so that I know my call wasn't the problem.
60. As an agent, I want a mistake in my call (unknown column, wrong column type, invalid option, number out of range) to raise an exception that says what's wrong, what's allowed and how to fix it, so that I can correct the call in one step.
61. As an agent, I want "did you mean" suggestions for misspelled column names, including differences only in case, so that I fix typos quickly.
62. As an agent, I want passing a Series, a NumPy array or a polars DataFrame to raise an error that shows the conversion, so that I don't have to guess.
63. As an agent, I want every exception from the package to share one base class and carry a stable code, so that I can catch them and branch on them.
64. As an agent, I want data problems to come back in the result and never as Python warnings, so that nothing important goes only to stderr.
65. As an agent, I want numerical warnings raised inside an analysis, such as division by zero, recorded as limitations, so that I see them next to the numbers they affect.
66. As an agent, I want a DataFrame with duplicated or colliding column labels to raise an error that names them and shows how to rename, so that column references stay unambiguous.

### Projects and masking

67. As a data scientist, I want to build a Project in Python code, so that I don't need a file.
68. As an agent that runs each script in a new process, I want to save the Project to a TOML file once and load it in later scripts, so that I don't redefine it every time.
69. As an agent, I want a Project file with an unknown or misspelled key to fail with an error listing the valid keys and the closest match, so that a typo can't silently switch a setting off.
70. As an agent handling personal data, I want to mark columns as sensitive in the Project's data policy, so that I can opt into masking their values.
71. As an agent, I want masking off by default even when sensitive columns are declared, so that nothing is hidden unless I ask.
72. As an agent, I want to switch masking on for every call through the Project, on for one call with `mask_sensitive=True`, or off for one call with `mask_sensitive=False`, so that I control what reaches my model provider.
73. As an agent, I want masked values replaced by placeholders that keep their counts and shares, with numbered placeholders for category levels, so that I keep the statistics without the raw values.
74. As an agent, I want masking to cover the printed summary, `to_dict()`, `to_json()` and `to_frame()` alike, so that values can't leak through another route.
75. As an agent, I want each result to state whether masking was applied and to which columns, so that I know what I'm looking at.
76. As an agent, I want `mask_sensitive=True` without a Project that declares sensitive columns to raise an error, so that I never believe data is masked when it isn't.
77. As an agent, I want `describe_analysis()` to tell me whether an analysis can show raw values and supports masking, so that I know where masking matters.
78. As a maintainer, I want the docs to call masking a guardrail against accidental exposure, not anonymization or access control, so that nobody relies on it for compliance.

### First-look analyses

79. As an agent, I want `describe_dataset` to report rows, columns, dtypes, missing and distinct counts and memory per column, so that I know the data's shape before anything else.
80. As an agent, I want `describe_dataset` to flag duplicated column labels, a non-unique index, empty data, all-missing columns and constant columns, so that structural problems surface first.
81. As an agent, I want `describe_dataset` to record a source and version only when I supply them and never to infer them, so that provenance claims stay honest.
82. As an agent, I want `infer_column_semantics` to rank likely logical types per column with evidence such as parse shares, distinct counts and uniqueness, so that I don't treat IDs or postal codes as numbers, or dates stored as text as categories.
83. As an agent, I want flags such as identifier-like, constant, numbers stored as text, dates stored as text, boolean-like and mixed value types, so that I know which columns need attention.
84. As an agent, I want inference never to override a schema I declared and to report where it disagrees, so that my declarations stay authoritative.
85. As an agent, I want date detection never to guess day/month order, so that ambiguous dates aren't silently misread.
86. As an agent, I want `validate_column_types` to check columns against declared logical types and count compatible, convertible, invalid and missing values per column, so that I find type problems without the package changing my data.
87. As an agent, I want invalid values shown with example row IDs and up to five example values, so that I can see why they fail.
88. As an agent, I want a declared column that's missing from the data reported as a violation instead of an exception, so that checking a broken file still completes.
89. As an agent, I want missing values in a column declared non-nullable reported as violations, so that I catch broken contracts.
90. As an agent, I want `profile_missing_values` to count native missing values and my declared sentinel values per column and per row without double counting, so that I know the data's true completeness.
91. As an agent, I want missing shares computed over all rows, plus how many values each row is missing, so that I see whether gaps cluster in some rows.
92. As an agent, I want a declared sentinel value that can never match its column's type to raise an error naming the likely intended value, so that "-999" given as text for an integer column doesn't silently count nothing.
93. As an agent, I want `detect_missing_sentinels` to flag likely stand-in values such as -999, 9999, "N/A", empty or blank strings and 1900-01-01, with evidence and example row IDs, so that I can decide which to declare as missing.
94. As an agent, I want suspected sentinel values kept apart from the ones I declared, and my data left unchanged, so that legitimate zeros, negatives and rare codes stay valid until I confirm otherwise.
95. As an agent, I want `summarize_numeric` to report counts, missing and infinite counts, mean, standard deviation, minimum, maximum, quantiles, median, trimmed mean, raw and scaled MAD, skewness and excess kurtosis per column, so that I understand each distribution in one call.
96. As an agent, I want each statistic's definition stated (sample standard deviation with ddof 1, bias-corrected skewness and kurtosis, the quantile method), so that I can compare with other tools.
97. As an agent, I want statistics that don't exist for constant or very small columns returned as null with the reason, so that I don't mistake a NaN for a number.
98. As an agent, I want `summarize_categorical` to report level counts and shares with their stated denominator, missing counts, cardinality, the most common levels and how many were left out, so that I understand each categorical column.
99. As an agent, I want labels that differ only by case or surrounding spaces ("Yes", "yes ", "YES") flagged as suspected inconsistencies, so that I clean them before modelling.
100. As an agent, I want unused categories of a pandas categorical reported, so that I notice levels that never occur.
101. As an agent, I want `find_duplicate_rows` to report duplicate groups, rows in them, excess copies and the duplicate share, with row IDs, so that I can judge and handle duplicates myself.
102. As an agent, I want the duplicate check to leave out my ID column by default and to state how missing values compare, so that unique IDs don't hide duplicated records.
103. As an agent, I want `find_candidate_keys` to find minimal unique single and composite keys within a search limit, with missing and duplicate counts, so that I know how rows are identified.
104. As an agent, I want the closest non-unique candidates ranked by uniqueness when no exact key exists, so that I spot a key broken by a few duplicates.
105. As an agent, I want the key search to say whether it was exhaustive, so that I don't mistake a cut-off search for proof that no key exists.
106. As an agent, I want `profile_target` to require the task instead of guessing it from the dtype, so that a numeric class label isn't mistaken for a regression target.
107. As an agent, I want class counts and shares, the imbalance ratio, missing labels, labels outside my expected set and, when I give a threshold, rare classes, so that I know the target before modelling.
108. As an agent, I want a single-class target, a binary task with other than two classes, a constant regression target and a classification target whose values are all distinct flagged, so that I catch a wrong target or task before training.
109. As an agent, I want `compute_association_matrix` to measure the association of every suitable pair of numeric and categorical columns, recording the method per pair, so that I see how variables relate in one call.
110. As an agent, I want each pair computed on its own complete rows, with the pair count reported, so that gaps in one column don't shrink every pair.
111. As an agent, I want directional measures marked with their direction, so that I don't read a correlation ratio as symmetric.
112. As an agent, I want the strongest associations listed first and perfect associations flagged as possibly duplicated or derived columns, so that I spot redundancy quickly.
113. As an agent, I want skipped columns (datetimes, all-missing, too many categories) listed with reasons, so that I know what the matrix doesn't cover.
114. As an agent, I want numeric columns that I list as categorical treated as categorical, so that coded categories such as 1, 2 and 3 aren't correlated as numbers.

### Speed

115. As an agent, I want each first-look analysis to finish in under about 5 seconds on 1 million rows and 100 mixed columns on a 2-vCPU, 8 GB machine, so that EDA fits inside an AgentCore session.
116. As an agent, I want analyses whose cost grows faster than the row count to have explicit work limits, so that a single call can't run for an hour.

### Quality and maintenance

117. As a maintainer, I want every statistic checked against SciPy, pandas or published worked examples, so that NumPy shortcuts can't drift from the correct values.
118. As a maintainer, I want property-based tests over generated DataFrames covering empty data, all-missing, constant and single-row columns, ties, infinities, mixed types, nullable and pyarrow dtypes, so that edge cases are covered systematically.
119. As a maintainer, I want a test that holds every registered analysis to the contract (complete docstring, running examples, strict-JSON results, determinism, no leaked warnings, unchanged input), so that every new analysis meets it automatically.
120. As a maintainer, I want CI on Python 3.12, 3.13 and 3.14 with both minimum and latest dependency versions, and on Linux ARM64, so that the environments AgentCore users run are tested.
121. As a maintainer, I want CI to build the wheel and import it from `sys.path` in a fresh interpreter, so that the offline route for sandboxed agents keeps working.
122. As a maintainer, I want deprecation and future warnings from pandas or NumPy to fail the test suite, so that code gets updated before it breaks.
123. As a contributor, person or coding agent, I want the catalogue entries of shipped analyses marked as implemented, so that I know which entries still describe future work.

## Implementation Decisions

### Package and build

- One distribution, `agent-eda`, with the import package `agent_eda`, built by the uv build backend from a src layout. The wheel is pure Python (`py3-none-any`) and ships a `py.typed` marker.
- Required dependencies are exactly the six in the project file, at their current minimums (ADR 0005). This slice needs no optional extras.
- Hypothesis and pyarrow join the development dependencies, pyarrow so that tests can cover pyarrow-backed columns. The benchmark harness is the implementer's choice.
- Any bundled data loads through `importlib.resources`, and nothing reads package files through `__file__`, so the wheel still imports from a zip on `sys.path` (ADR 0005).
- `agent_eda.__version__` gives the version, also when the package is imported from the wheel file.

### Public API layout

- Every analysis and discovery function is an attribute of the top-level package, for example `ae.summarize_numeric`. The top level also exports `Project`, `DataPolicy`, `ColumnSpec`, the `LogicalType` literal, the `AnalysisResult` base class and `__version__`. Exceptions live in a public `errors` module and result classes in a public `results` module.
- Analyses live in submodules that follow the catalogue's grouping: `data.provenance` (describe_dataset), `data.validation` (infer_column_semantics, validate_column_types), `data.structure` (find_candidate_keys), `data.duplicates` (find_duplicate_rows), `data.missingness` (profile_missing_values, detect_missing_sentinels), `data.targets` (profile_target), `statistics.descriptive` (summarize_numeric, summarize_categorical) and `statistics.association` (compute_association_matrix). Analysis names are unique across the package and match the catalogue.
- Loading is lazy. Importing `agent_eda` or calling a discovery function never imports SciPy, statsmodels or scikit-learn; an analysis imports what it needs when it runs. numpy, pandas and pydantic may load on import.
- Type checkers and editors see every top-level export statically, and `dir(agent_eda)` lists them, so completion works despite lazy loading.
- A registration decorator records each analysis's metadata (summary, tags, parameters, methods, declared statuses, declared finding codes, result class, masking support, work-limit parameters) and sets `__test__ = False` on the function.

### Calling convention

- `data`, a pandas DataFrame, is the only positional parameter. Everything else is keyword-only.
- Parameters that take column names end in `column` or `columns`: `columns`, `id_column`, `target_column`, `categorical_columns`.
- `columns=None` selects every suitable column for the analysis, in DataFrame order. An explicit list keeps the caller's order. An empty list, a name listed twice, or a column the analysis can't handle raises.
- Fixed choices are `Literal` types. Numeric arguments have documented valid ranges, checked before any computation.
- Defaults exist only for standard conventions and work limits, and the result's parameters record them: quantile levels, quantile method, trim fraction, `top_n`, `max_key_size`, `max_combinations`, `max_categories`, `max_pairs`. Project-specific inputs (schemas, sentinel values, expected labels, the rare-class share) have no defaults; when they're left out, the checks that need them don't run.
- `data` is never modified. Values, dtypes, index, column order and `attrs` stay identical.
- Input checks run before any computation. A Series, a NumPy array or a polars DataFrame raises with the one-line conversion. Duplicated column labels, labels whose string forms collide, and MultiIndex columns raise with a renaming hint; describe_dataset is the exception and reports them as findings.
- Accepted column dtypes: NumPy numerics, bools, datetimes (naive and time-zone aware) and timedeltas; pandas nullable dtypes (`Int64`, `Float64`, `boolean`, `string`); pandas 3's default string dtype; categoricals; object columns; pyarrow-backed dtypes.

### Analysis results

- `AnalysisResult` is a frozen pydantic model. Its shared fields are reserved names: `analysis`, `method` (null when the analysis has no methods), `status`, `parameters` (resolved arguments with defaults and the resolved column list; the DataFrame appears only as its row and column counts), `support` (rows in the data, columns selected, and per column the eligible, evaluated and excluded counts with reasons), `findings`, `limitations`, `execution` (the versions of agent-eda, Python, numpy, pandas and any other library the analysis imported) and, on analyses that support masking, `masking`.
- Each analysis has its own result class named after it (summarize_numeric returns `SummarizeNumericResult`), with its typed fields directly on the result. Multi-column analyses keep one entry per column, keyed by column name, each with its own status and, when it didn't complete, the reason.
- Results hold no timestamps or durations. Output order is fixed (DataFrame or caller column order, then explicit tie-breaks) and never follows hash or set iteration order, so identical calls in the same environment give byte-identical JSON across processes.
- Fractions are named `…_share` and lie between 0 and 1. Counts are integers.
- A value that doesn't exist for the data is null, and a limitation names the field path and the reason. NaN and infinity never appear in the record.
- Any list in the record that grows with the data is capped and carries its total: 10 row IDs, 5 example values, and 100 entries for other lists such as levels, groups, keys, pairs and sentinel candidates. The complete tables stay on the result object outside the serialized record, and `to_frame()` returns them.
- `to_frame()` returns the analysis's main table. Results with more than one table accept a `table` literal, and the printed summary names the tables.
- `to_dict()` returns the record as JSON-compatible Python objects. `to_json()` returns strict JSON with non-finite numbers forbidden. Row IDs convert deterministically: timestamps to ISO 8601 strings, MultiIndex tuples to lists, NumPy scalars to Python numbers.
- `str()` and `repr()` give the same plain text from a fixed template: a first line with analysis, status and data size, the key table (at most 10 rows), findings (at most 5), limitations (at most 5), and a footer on how to get the rest. Hard caps are 40 lines and 120 characters per line, and anything left out is counted. Numbers show 4 significant digits and counts use thousands separators. There is no HTML display.

### Statuses

- The eight statuses: `completed`, `partial`, `insufficient_data`, `not_applicable`, `missing_prerequisites`, `budget_exceeded`, `blocked_by_policy`, `failed`. Each analysis declares the ones it can return, and a result never carries an undeclared status.
- Per-column entries use the same vocabulary. The overall status is `completed` when every selected column completed and `partial` when some completed and some didn't. When none completed, it is the status the columns share, or, if they differ, the first of `failed`, `insufficient_data` and `not_applicable` that occurs.
- `failed` covers anticipated numerical failures and carries an error code and message. Unexpected exceptions, which mean a bug, propagate instead of becoming results.
- No analysis in this slice returns `missing_prerequisites` or `blocked_by_policy`.

### Findings and limitations

- A finding has a stable code, a one-sentence claim from a fixed template, an assessment, the columns it concerns, and the result field paths behind it.
- Assessments: `descriptive` for neutral facts; `suspected` for possible problems someone needs to judge; `violation` when the data breaks something declared (a schema, an expected label set, the task); `passed` when a declared check held; `unknown` when a declared check couldn't be evaluated.
- Each analysis declares its finding codes. Limitation codes come from one package-wide list. Both are public API, so renaming one follows the rule for renamed names.

### Row IDs

- validate_column_types, detect_missing_sentinels, find_duplicate_rows and profile_target point at rows and accept `id_column`. Row IDs are index labels by default and values of `id_column` when given. Those analyses leave the ID column out of their default column selection.
- When the index or ID column has duplicates, the result carries an `ambiguous_row_ids` limitation recommending a unique `id_column`.

### Errors

- One base class, `AgentEDAError`, with a stable `code`. Each subclass also inherits the closest built-in exception so generic handlers work: `InvalidArgumentError` (ValueError), `ColumnNotFoundError` (LookupError), `ColumnTypeError` (TypeError), `UnknownAnalysisError` (LookupError), `ProjectFileError` (ValueError).
- Every message says what's wrong, what's allowed and how to fix it. An unknown column lists the available columns and the closest matches, case-insensitive matches first. An invalid literal lists the allowed values and the closest one. A wrong column type names the dtype and the analysis that fits. An unsupported input gives the conversion call.
- Validation errors from pydantic are re-raised as `InvalidArgumentError` in that format. pydantic's raw errors never reach the caller.

### Warnings

- Data problems never become Python warnings. Analyses capture warnings: numerical RuntimeWarnings (division by zero, invalid values, overflow, precision loss) become limitations, while DeprecationWarning and FutureWarning are suppressed at runtime and fail the test suite. The package calls `warnings.warn` only for renamed names, and nothing is renamed yet.

### Discovery

- `ae.guide()` prints one page of plain text, at most about 60 lines, and returns nothing. It covers what the package is, the call pattern, how to read results and statuses, a reminder to print results, every analysis grouped by module with its one-line summary (generated from the registry), and the first-look order: describe_dataset, infer_column_semantics and validate_column_types, profile_missing_values, detect_missing_sentinels, summarize_numeric and summarize_categorical, find_duplicate_rows and find_candidate_keys, profile_target, compute_association_matrix.
- `ae.list_analyses()` returns a typed listing (name, module, summary, masking support) that prints compactly and has `to_frame()`.
- `ae.search_analyses(query, limit=5)` ranks analyses by keyword matches against names, summaries, tags and when-to-use text, expanded with a small curated synonym list. Scoring is deterministic and ties go to the alphabetically first name. It returns a typed listing with each match's summary and the terms that matched. No match gives an empty listing that points to the guide. No embeddings, no network.
- `ae.describe_analysis(name)` takes the short or the fully qualified name and returns a typed description: the signature; each parameter's type, default, allowed values and description from the docstring's `Args:`; declared statuses with their reasons; finding codes; result fields and their meanings; masking support; work-limit parameters; and the example. Its record includes a JSON Schema of the non-data parameters. An unknown name raises `UnknownAnalysisError` with suggestions.

### Docstrings

- Google style, with sections in this order: a summary line phrased as a question and ending in "?"; `When to use:` with alternatives; `Args:` covering every parameter with type, unit, allowed values and the meaning of None; `Returns:` naming the fields to read, how to interpret them and the finding codes; `Statuses:` listing each declared status and why; `Examples:` as a doctest that prints a result; `See Also:` naming existing analyses.
- Search and the guide read the summary line and the when-to-use text from the docstring, so there is no second copy to keep in sync.

### Project, data policy and masking

- `Project` is a frozen pydantic model that rejects unknown fields. It has an optional `name` and a `data_policy`. `DataPolicy` has `sensitive_columns` (default none) and `mask_sensitive` (default false); setting `mask_sensitive` with no sensitive columns raises. Dataset roles are not part of this slice.
- `Project.load(path)` reads TOML with the standard library. The file has `name` at the top level and a `data_policy` table that mirrors the fields. Unknown keys raise `ProjectFileError` with the valid keys and the closest match, and TOML syntax errors report the line. `project.save(path, overwrite=False)` writes the same format deterministically without a TOML-writing dependency and refuses to overwrite unless told to.
- The analyses whose results can contain raw values accept `project` and `mask_sensitive`: validate_column_types, detect_missing_sentinels, summarize_categorical, find_duplicate_rows and profile_target. The other six never show raw values and don't take them.
- Resolution: a per-call `mask_sensitive` of True or False wins; otherwise the Project's setting applies; without a Project nothing is masked. A per-call True without a Project that declares sensitive columns raises `InvalidArgumentError`.
- Within sensitive columns, masking replaces category levels and labels, example values, sentinel value candidates, target class labels, and row IDs taken from a sensitive `id_column`. The replacements are numbered placeholders per column in result order (`<masked 1>`, `<masked 2>` and so on) that reveal neither content nor length. Counts, shares and numeric statistics stay. Index labels are never masked; the docs say to pass a non-sensitive `id_column` when the index holds sensitive values.
- Masking covers the text display, `to_dict()`, `to_json()` and `to_frame()` alike. The result's `masking` field records whether it was applied and to which columns.
- The docs call masking a guardrail against accidental exposure, not anonymization and not access control (ADR 0001).

### The first-look analyses

Each analysis below lists the question its docstring summary asks, its parameters beyond `data`, what it returns, its findings and its declared statuses. Every analysis can also return `failed`.

**describe_dataset**: "What does this dataset contain?"

- Parameters: optional `source` and `source_version` strings.
- Returns row and column counts; per column the dtype as pandas reports it, the non-missing, missing and distinct counts and the memory in bytes; total memory; facts about the index (dtype, uniqueness, missing labels); and the supplied provenance, where anything not supplied is recorded as not supplied and never inferred.
- Findings: duplicated or colliding column labels, MultiIndex columns, a non-unique index, empty data, all-missing columns, constant columns.
- Statuses: `completed`, also for empty data.

**infer_column_semantics**: "What kind of values does each column hold?"

- Parameters: `columns`; optional `declared_schema` mapping column names to a logical type or a `ColumnSpec`.
- Returns per column the candidate logical types ranked by a heuristic score between 0 and 1, with reasons; flags (`identifier_like`, `constant`, `all_missing`, `numbers_as_text`, `dates_as_text`, `non_iso_date_text`, `boolean_like`, `mixed_value_types`); and the evidence: non-missing and distinct counts, uniqueness share, and the shares of values that parse as numbers, integers, ISO 8601 dates and booleans.
- Scores come from documented, deterministic rules and are not probabilities. A declared type is never overwritten; when the top candidate differs, the finding is suspected.
- Dates parse as ISO 8601 only. Other date-like text gets a flag and is never parsed, so day/month order is never guessed. Parse checks run on distinct values weighted by their counts.
- Must classify correctly: postal codes with leading zeros stored as text; numeric-looking IDs; ISO date strings; yes/no and 0/1 columns; floats whose values are all whole numbers apart from gaps (integer); free-text sentences (text); a few labels repeated over many rows (categorical); an all-distinct integer sequence (flagged `identifier_like`).
- Statuses: `completed`, `partial`, `insufficient_data`, `not_applicable`.

**validate_column_types**: "Do the columns hold the logical types I declared?"

- Parameters: required `schema` mapping column names to a logical type or a `ColumnSpec` (logical type, `nullable`, optional `datetime_format`); `id_column`; `project`; `mask_sensitive`.
- Returns per declared column the expected logical type, the observed dtype, whether the column is present, and counts of compatible values (already stored as that type), convertible values (they convert exactly without changing meaning, such as "42" or 42.0 for integer, or text in ISO 8601 or the declared datetime format), invalid values and missing values. Invalid values come with up to 10 example row IDs and 5 distinct example values. Undeclared columns are listed.
- Findings: violation for invalid values, for a declared column missing from the data, and for missing values in a non-nullable column; passed for a clean column; unknown for a column without values to check. Nothing is coerced.
- Statuses: `completed`.

**profile_missing_values**: "How much data is missing, and where?"

- Parameters: `columns`; optional `sentinel_values` mapping column names to declared sentinel values.
- Returns per column the native missing count (None, NaN, NaT, pd.NA), the sentinel count, the combined missing count without double counting, and the missing share over all rows; per dataset the number of complete rows and the distribution of how many selected values each row is missing. Empty strings count as missing only when declared.
- A declared sentinel value that can never match its column's type raises `InvalidArgumentError` naming the likely intended value.
- Findings: all-missing columns, declared sentinel values present.
- Statuses: `completed`, `insufficient_data` (no rows: counts are zero and shares null), `not_applicable`.

**detect_missing_sentinels**: "Which values look like stand-ins for missing data?"

- Parameters: `columns`; optional declared `sentinel_values`; `id_column`; `project`; `mask_sensitive`.
- Candidates come from fixed, documented rules with no tunable thresholds: values from a built-in list, and repeated numeric values at the column's minimum or maximum whose distance to the nearest other value exceeds the range of all other values. The built-in list covers numeric codes (-9, -99, -999, -9999, 99, 999, 9999, 99999, 999999) when they sit at the column's minimum or maximum; text, compared case-insensitively ("", whitespace only, "NA", "N/A", "NULL", "None", "nan", "-", "?", "unknown", "missing"); and dates (1900-01-01, 1970-01-01, 2099-12-31, 9999-12-31).
- Returns each candidate's column, value, count, share, evidence codes, up to 10 example row IDs and confirmation: declared or suspected. Declared values give descriptive findings and built-in matches give suspected ones. Data is never changed.
- Statuses: `completed`, `insufficient_data`, `not_applicable`.

**summarize_numeric**: "How are the values in each numeric column distributed?"

- Parameters: `columns`; `quantiles` (default 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99); `quantile_method` (NumPy's quantile methods, default linear); `trim_fraction` (default 0.1, from 0 up to but excluding 0.5).
- Returns per column: count, missing, infinite, zero and negative counts; mean; standard deviation with ddof 1; minimum and maximum; quantiles; median; trimmed mean with `trim_fraction` cut from each end; raw MAD and MAD scaled for consistency with the normal distribution (×1.4826); bias-corrected skewness; bias-corrected excess kurtosis. Moments use finite values only.
- Null with a reason: standard deviation with fewer than 2 values; skewness with fewer than 3 values or zero variance; kurtosis with fewer than 4 values or zero variance.
- `columns=None` selects numeric columns except booleans. A boolean column requested by name is read as 0 and 1, with a limitation. Integers beyond 2^53 add a float-precision limitation. Weights are not supported in this slice.
- Findings: constant columns, infinite values present.
- Statuses: `completed`, `partial`, `insufficient_data`, `not_applicable`.

**summarize_categorical**: "Which values occur in each categorical column, and how often?"

- Parameters: `columns`; `top_n` (default 10, for display only); `project`; `mask_sensitive`.
- Returns per column the non-missing and missing counts, cardinality, levels with counts and shares (the denominator is the non-missing count, and the result states it), the most common levels, the number of levels left out, the number of levels seen once, unused categories of a categorical dtype, and whether values of different Python types are mixed. Order is count descending, then label. Labels are kept exactly as stored.
- Findings: labels that differ only by case or surrounding spaces (suspected), unused categories (descriptive), mixed value types (suspected).
- `columns=None` selects object, string, categorical and boolean columns. A numeric column requested by name is summarized by its values.
- Statuses: `completed`, `partial`, `insufficient_data`, `not_applicable`.

**find_duplicate_rows**: "Are there exact duplicate rows?"

- Parameters: `columns` (default every column except `id_column`); `id_column`; `project`; `mask_sensitive`.
- Returns duplicate groups (size and row IDs, ordered by first occurrence), the group count, rows in duplicate groups, excess copies (rows in groups minus groups), the duplicate share (excess copies over all rows) and the compared columns. Missing values compare equal, as in pandas, and the result says so. No rows are removed. `to_frame()` lists every row in a duplicate group with its group number.
- Findings: duplicates present (suspected), none found (descriptive).
- Statuses: `completed`, `insufficient_data`, `not_applicable`.

**find_candidate_keys**: "Which columns, alone or combined, uniquely identify rows?"

- Parameters: `columns`; `max_key_size` (default 2); `max_combinations` (default 10,000).
- Searches single columns, then combinations in increasing size, keeping only minimal keys: supersets of a found key are skipped. Each candidate reports its columns, distinct key count, duplicate rows, rows with a missing key value, uniqueness share and whether it is minimal. Keys with missing values are listed separately and are not primary keys. Without an exact key, the 10 most unique column sets are listed.
- The result says whether the search was exhaustive. Hitting `max_combinations` stops the search with `budget_exceeded`, keeping what was found and marking it non-exhaustive.
- A deterministic, evenly spaced subset of rows may reject a candidate cheaply, because a duplicate in the subset proves non-uniqueness. Uniqueness is only ever confirmed on all rows.
- Statuses: `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`.

**profile_target**: "What does the target look like for this task?"

- Parameters: required `target_column`; required `task` (`binary_classification`, `multiclass_classification` or `regression`); optional `expected_labels`; optional `rare_share` (between 0 and 1); `id_column`; `project`; `mask_sensitive`.
- Classification returns class counts and shares over valid labels, the number of classes, majority and minority shares, the imbalance ratio, missing labels, labels outside `expected_labels` with counts and example row IDs, and rare classes only when `rare_share` is given. Regression returns summarize_numeric's fields for the target (one shared implementation) plus the count of non-finite values.
- Findings: violation for a single class, a binary task with other than two classes, labels outside `expected_labels` and a constant regression target; suspected for a classification target whose values are all distinct; descriptive for missing labels and rare classes.
- A regression task on a non-numeric column raises `ColumnTypeError` suggesting a classification task or a conversion. The task is never inferred.
- Statuses: `completed`, `insufficient_data`.

**compute_association_matrix**: "How strongly are pairs of columns associated?"

- Parameters: `columns`; `categorical_columns` (numeric columns to treat as categorical); `numeric_method` (`pearson`, `spearman` or `kendall`, default `pearson`); `categorical_method` (`cramers_v` or `cramers_v_bias_corrected`, default `cramers_v_bias_corrected`); `max_categories` (default 50); `max_pairs` (default 5,000).
- Column types: numeric dtypes other than booleans are numeric unless listed in `categorical_columns`; booleans, strings, objects and categoricals are categorical. Datetime and timedelta columns, all-missing columns and categorical columns with more levels than `max_categories` are skipped with a reason.
- Measures: the numeric method for numeric pairs, where Spearman and Kendall rank each pair's own complete rows and Kendall is tau-b; Cramér's V for categorical pairs, bias-corrected after Bergsma (2013) by default; the correlation ratio (eta) for categorical-numeric pairs, marked as directional from the categorical to the numeric column.
- Each pair uses its own complete rows and reports its pair count. Undefined pairs (a constant column, fewer than 3 complete rows, a single level) are null with a reason. No p-values or intervals.
- The main table has one row per pair: both columns, their types, method, value, pair count, direction and undefined reason. A second table gives the wide matrix. The record keeps the 100 strongest pairs by absolute value. Findings flag perfect associations (absolute value 1 within floating-point tolerance) as suspected duplicated or derived columns.
- More pairs than `max_pairs` gives `budget_exceeded` without computing anything, stating the pair count and how to narrow `columns` or raise the limit. Fewer than two eligible columns is `not_applicable`.
- Statuses: `completed`, `budget_exceeded`, `insufficient_data`, `not_applicable`.

### Performance

- NumPy first (ADR 0002): convert each column to a NumPy array once, vectorize, and work on distinct values with counts for text columns. pandas is used for grouping, hashing, nullable types, categoricals and time zones.
- Target: with default arguments, each analysis finishes in under 5 seconds on a deterministic benchmark DataFrame of 1,000,000 rows and 100 mixed columns (floats with missing and infinite values, nullable integers, strings and categoricals with cardinality from 2 up to all-distinct, booleans, datetimes, about 5% missing overall) on a 2-vCPU, 8 GB machine, the size of an AgentCore session.
- No threads or processes beyond what NumPy uses internally.

### Compatibility

- The code supports the minimum versions in the project file through the latest releases: numpy 1.26 through 2.x without NumPy-2-only functions, pandas 2.2 through 3.x including copy-on-write and the new default string dtype, scikit-learn 1.5 and later, scipy 1.11.2 and later, statsmodels 0.14.1 and later, and pydantic 2.7 and later, on Python 3.12, 3.13 and 3.14.

### Docs and records

- The README's planned usage becomes real usage, and its status line says what 0.1 contains.
- The catalogue entries of the 11 analyses are marked implemented, with their differences from the catalogue noted: no `context` argument, literal methods, defaults in compute_association_matrix instead of required type and method mappings, `max_key_size` defaulting to 2, weights deferred, "convertible" instead of "parseable", and shares instead of rates.
- GLOSSARY.md now defines Logical type, Sentinel value, Support, Sensitive column and Masking; they were added together with this spec.

## Testing Decisions

### One seam

All tests go through one seam: the public `agent_eda` API, used the way callers use it. Import the package, call an analysis or discovery function with a DataFrame and keyword arguments, and check what comes back. Tests never import private modules or assert on internal helpers. The NumPy calculations, the text renderer and the registry are covered through the analyses and discovery functions that use them.

### What makes a good test here

- It names one external behaviour and checks it through the public API: result fields, status, findings, printed text, `to_frame()`, `to_dict()`, `to_json()`, or the exception and its message.
- It uses a hand-built DataFrame small enough that the right answer is obvious, or generated DataFrames for properties.
- Reference values are computed independently on the same data (SciPy, pandas, NumPy or a published worked example) and compared within a stated tolerance.
- No network, no unseeded randomness, no dependence on test order.
- Test function names never reuse an analysis name.

### Test groups

1. **Behaviour.** Per analysis, the canonical cases and edge rules listed under it above.
2. **Reference values.** summarize_numeric against SciPy (`skew` and `kurtosis` with `bias=False`, `trim_mean`, `median_abs_deviation`) and NumPy quantiles for every method. Association measures against SciPy (`pearsonr`, `spearmanr`, `kendalltau`, contingency `association`) and published examples for bias-corrected Cramér's V and the correlation ratio. Counts against pandas.
3. **Contract, looping over `ae.list_analyses()`.** Docstring sections present and in order; summary line ends in "?"; every parameter documented and nothing extra; See Also names resolve; doctest examples run; `__test__` is False; `describe_analysis()` works; searching for the analysis's own name returns it first; results only use declared statuses and finding codes.
4. **Properties, with Hypothesis.** A mixed-dtype DataFrame strategy: floats with NaN and infinity, nullable integers, bool and boolean, object strings including empty and whitespace-only, pandas 3 strings, categoricals with unused categories, naive and aware datetimes, pyarrow-backed columns; zero rows, one row, zero columns, constant and all-missing columns. For every analysis: no unexpected exception on valid input; status within the declared set; `to_json()` is strict JSON; two identical calls give identical JSON; the input DataFrame is unchanged (values, dtypes, index, columns); no warning escapes; the printed text respects the 40-line and 120-character caps; with masking on, no raw value from a sensitive column appears in the printed text, `to_json()` or `to_frame()`.
5. **Discovery.** The guide names every analysis and stays within its size. Search returns the expected first result for each query in the table below. Unknown analysis names get close suggestions.
6. **Errors.** Every error code, its exception class, and the message content: available columns, closest match, allowed values, conversion call.
7. **Project.** Built in code; save and load round trip; unknown and misspelled keys; a TOML syntax error reports its line; masking resolution, including the error for masking without sensitive columns.
8. **Packaging, in fresh subprocesses.** `import agent_eda` and each discovery function leave SciPy, statsmodels and scikit-learn unimported. The built wheel is pure Python, imports from `sys.path` in an isolated interpreter, runs one analysis and prints it. The same call gives identical JSON under two different `PYTHONHASHSEED` values.
9. **Benchmark.** The deterministic 1M × 100 dataset and one timing per analysis with default arguments, marked so that the default test run skips it.

Search acceptance cases. The queries use an agent's wording, not the glossary's, on purpose:

| Query | Expected first result |
| --- | --- |
| missing values | profile_missing_values |
| nulls per column | profile_missing_values |
| -999 placeholder codes | detect_missing_sentinels |
| duplicate rows | find_duplicate_rows |
| primary key | find_candidate_keys |
| correlation | compute_association_matrix |
| check data types against a schema | validate_column_types |
| what kind of column is this | infer_column_semantics |
| class imbalance | profile_target |
| mean and standard deviation | summarize_numeric |
| value counts | summarize_categorical |
| dataset overview | describe_dataset |

### Warning policy

DeprecationWarning and FutureWarning raised while the package's code runs fail the suite. The contract tests check separately that no warning escapes an analysis.

### What gets tested

The whole public API, through the seam: the 11 analyses, the four discovery functions, `Project`, `DataPolicy`, `ColumnSpec`, the result classes and the exceptions.

### Prior art

None yet. pytest is configured with strict markers and strict config, coverage settings exist, and pre-commit runs pytest before each push, but there are no tests. The fixtures, the mixed-dtype Hypothesis strategy and the registry-driven contract tests written here become the pattern for later slices.

### CI

A GitHub Actions workflow, ready to run once the repo is on GitHub:

- lint (`ruff check`, `ruff format --check`) and `ty`;
- tests with the latest dependencies on Python 3.12, 3.13 and 3.14, Linux x86_64;
- tests with the lowest direct dependencies on Python 3.12 (numpy 1.26 has no 3.13 wheels);
- tests on Linux ARM64, Python 3.12, latest dependencies;
- packaging: build the wheel, check that it is pure Python, run the `sys.path` import test, install it with `--no-deps` into an environment holding only the minimum required versions, and run a smoke test;
- the benchmark, on manual trigger only.

## Out of Scope

- Every catalogue analysis beyond these 11, including value-domain, unit, precision and datetime or time-zone validation, missingness patterns, and association tests with p-values.
- Weighted summaries in summarize_numeric, summarize_categorical and profile_target.
- Dataset roles, role checks, holdout guardrails and data policy permissions. No analysis here returns `blocked_by_policy` or `missing_prerequisites`.
- Statistical hypothesis tests, p-values, confidence intervals and hypothesis families; randomized methods and seeds.
- Optional dependency groups and the missing-dependency error.
- Generated agent docs: `llms.txt`, `SKILL.md`, paste-in agent instructions and a documentation site.
- Tool wrappers for Strands, MCP or other frameworks.
- Creating the GitHub repository, the PyPI release and trusted publishing.
- Agent test runs.
- Sampling data too large for memory.
- HTML display.
- Presets and orchestration, report assembly, an artifact store, S3 loading and cancellation.
- Masking numeric statistics, and anonymization in general.
- Deprecation aliases; nothing has been renamed yet.

## Further Notes

- The single test seam and this scope were proposed in the session before this spec was written, and the user didn't object.
- This spec extends ADR 0004: `project=` also appears on analyses whose results can show raw values, because the data policy's masking applies there. ADR 0004 now says "roles or the data policy".
- Weights are deferred because correct weighted statistics need a declared weight interpretation (frequency, probability or importance), which belongs with the sampling analyses.
- Still unverified about AgentCore: whether sandbox-mode Code Interpreter allows `pip install --no-deps` (the `sys.path` route covers the case where it doesn't), and the exact preinstalled versions, which come from one third-party report from March 2026. Code Interpreter returns only printed output, so every example prints its result.
- The repo has no git remote yet, so CI can't run until it's on GitHub. The workflow is still part of this slice.
- Likely next specs: generated agent docs and paste-in instructions; dataset roles and role checks; the 0.1 release; agent test runs; sampling for large data; the next catalogue analyses.
