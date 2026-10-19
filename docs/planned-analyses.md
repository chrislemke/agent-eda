# Planned analyses

The 11 shipped analyses are the first of about 280 planned. The tables below list the rest by area, each with the `agent_eda` submodule it is meant for. Names and modules are proposals and can change while an analysis is built. Once one ships, its code and docstring are the source of truth, and it moves from these tables to the [Analyses](https://github.com/chrislemke/agent-eda/blob/main/README.md#analyses) section of the README: into the first-look table if it is a step of `ae.guide()`'s first look, otherwise into the "Other analyses" table under its module.

Planned analyses follow the same rules as the shipped ones, which [AGENTS.md](../AGENTS.md) lists. They take the DataFrame and typed keyword arguments, with no shared run context ([ADR 0004](adr/0004-plain-functions-with-explicit-arguments.md)). They read local files only and open no network connection ([ADR 0006](adr/0006-local-mcp-server-over-stdio-data-by-path.md)). They ask for project knowledge, such as units, keys, time cutoffs or dataset roles, rather than guess it. They report p-values and intervals only where the data design supports them, and say so otherwise.

<details>
<summary>Shared functions (8)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `load_dataset` | `io` | Load an approved local tabular file, such as CSV, TSV or Parquet, select permitted columns and record the source and loading metadata. It opens no network connection. |
| `export_eda_json` | `io` | Write a validated UTF-8 JSON report to a local file and return its path and checksum. |
| `validate_project` | `project` | Validate and normalize a project's goals, target definition, metrics, restrictions, dataset roles and analysis settings; report missing or contradictory fields. |
| `check_analysis_requirements` | `discovery` | Check required data, metadata, permissions, dependencies, sample support and budget before running an analysis. |
| `run_standard_eda` | `execution` | Call the first-look analyses on one DataFrame with explicit keyword arguments and return their results, recording which ran, were skipped, were inapplicable or failed. It shares no run state between analyses. |
| `run_analysis` | `execution` | Call one registered analysis by name with explicit keyword arguments and return its result; arguments are checked before any computation starts. |
| `validate_analysis_result` | `reporting` | Check the result schema, execution status, evidence, limitations and strict JSON serialization, including non-finite numbers. |
| `assemble_eda_report` | `reporting` | Combine results into a structured JSON report containing findings, uncertainty, limitations, coverage and unresolved questions. |

</details>

<details>
<summary>Data quality and structure (46)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `trace_data_lineage` | `data.provenance` | Trace supplied source, transformation and output relationships; identify missing links or inconsistent versions. |
| `describe_collection_process` | `data.provenance` | Summarize how, when and from whom data was collected, including instruments, inclusion rules and known collection changes. |
| `validate_data_contract` | `data.provenance` | Check required columns, schema, keys and declarative data rules against the declared contract. |
| `audit_data_permissions` | `data.provenance` | Check declared permissions for reading, analysis, model use, evaluation, retention and external processing. |
| `validate_measurement_units` | `data.validation` | Check supplied units, dimensional compatibility and declared conversion rules; flag missing unit evidence. |
| `validate_value_domains` | `data.validation` | Check numeric bounds, allowed categories, formats and inclusive or exclusive value constraints. |
| `validate_numeric_precision` | `data.validation` | Find unexpected decimal precision, quantization, overflow risks and precision loss relative to declared requirements. |
| `validate_datetime_values` | `data.validation` | Check date parsing, formats, time zones, daylight-saving ambiguities and allowed temporal bounds. |
| `validate_business_rules` | `data.validation` | Evaluate declared cross-field business rules and return violation counts, rates and permitted examples. |
| `profile_observation_units` | `data.structure` | Check the declared meaning and granularity of each row and summarize observation units and repeated measurements. |
| `profile_entities` | `data.structure` | Count entities, records per entity, repeated observations and optional observation spans. |
| `profile_table_relationships` | `data.structure` | Measure declared relationship cardinalities, referential coverage and unmatched keys between tables. |
| `check_functional_dependencies` | `data.structure` | Check declared determinant-to-dependent column relationships and quantify conflicting mappings. |
| `find_near_duplicate_records` | `data.duplicates` | Find similar tabular records using declared numeric or categorical matching rules; return scores and candidate groups. |
| `find_duplicate_entities` | `data.duplicates` | Find repeated entity records and compare them with declared rules for legitimate repeated observations. |
| `find_conflicting_records` | `data.duplicates` | Find records sharing a key but disagreeing on selected values beyond declared tolerances. |
| `assess_duplicate_impact` | `data.duplicates` | Quantify how duplicate groups affect effective row counts, target summaries and group representation. |
| `sample_analysis_data` | `data.sampling` | Create a reproducible random, stratified, group or time-based analysis sample and retain inclusion information and weights where justified. |
| `compare_sample_to_population` | `data.sampling` | Compare sample coverage and selected distributions against a reference population, using supplied weights. |
| `profile_sampling_design` | `data.sampling` | Summarize declared strata, clusters, stages, inclusion probabilities and sampling coverage. |
| `audit_sample_inclusion` | `data.sampling` | Audit which eligible records were included or excluded and quantify associated selection patterns. |
| `validate_sampling_weights` | `data.sampling` | Check missing, invalid or extreme weights and consistency with supplied inclusion probabilities and totals. |
| `estimate_effective_sample_size` | `data.sampling` | Estimate effective observations using an explicitly selected weighting, clustering or serial-dependence method. |
| `assess_group_representation` | `data.sampling` | Report group coverage and underrepresentation relative to supplied reference data or declared support thresholds. |
| `analyze_missingness_patterns` | `data.missingness` | Summarize joint missingness patterns, co-occurrence and concentrations of incomplete records. |
| `test_missingness_associations` | `data.missingness` | Test associations between missing-value indicators and target, group, time or other observed variables. |
| `fit_missingness_diagnostic_model` | `data.missingness` | Evaluate how predictably a field is missing using observed variables and valid diagnostic evaluation. |
| `analyze_multioutput_targets` | `data.targets` | Summarize joint target distributions, dependencies, co-occurrence and valid multi-output support. |
| `audit_label_consistency` | `data.targets` | Find contradictory labels for equivalent observations and violations of supplied label rules. |
| `audit_label_provenance` | `data.targets` | Check label sources, generation procedures, annotators, revisions and traceability to source evidence. |
| `analyze_label_latency` | `data.targets` | Measure delays between observation, outcome occurrence and the time a label became available. |
| `assess_outcome_maturity` | `data.targets` | Determine which observations have complete outcome follow-up at the supplied cutoff and summarize immature labels. |
| `estimate_repeatability` | `data.measurement` | Estimate consistency of repeated measurements on the same subjects under the declared measurement design. |
| `estimate_interrater_agreement` | `data.measurement` | Estimate categorical or ordinal agreement using the selected kappa or other supported agreement measure. |
| `estimate_intraclass_correlation` | `data.measurement` | Estimate the explicitly selected intraclass correlation form and its uncertainty. |
| `estimate_measurement_agreement` | `data.measurement` | Estimate measurement bias and agreement limits for paired instruments, accounting for declared repeated subjects. |
| `compare_measurement_sources` | `data.measurement` | Compare measurement distributions or bias across instruments, sources or collection processes. |
| `analyze_measurement_error_sensitivity` | `data.measurement` | Re-run a selected analysis under explicitly supplied measurement-error scenarios and compare the conclusions. |
| `audit_join_integrity` | `data.pipeline` | Check join cardinality, unmatched keys, duplicate expansion, changed row counts and introduced nulls. |
| `audit_row_conservation` | `data.pipeline` | Quantify retained, removed, added or multiplied observations and compare them with declared pipeline behavior. |
| `audit_introduced_missingness` | `data.pipeline` | Find newly missing values in aligned observations and distinguish expected changes from unexplained null introduction. |
| `compare_dataset_schemas` | `data.pipeline` | Report added, removed or changed columns and logical types against an approved reference schema. |
| `verify_input_versions` | `data.pipeline` | Compare supplied dataset, settings, code, dependency and model identities and checksums. |
| `verify_reproducibility` | `data.pipeline` | Compare recorded executions, seeds, environments, outputs and numerical results under declared exact or tolerance-based criteria. |
| `audit_numeric_approximations` | `data.pipeline` | Record approximation settings and error guarantees and compare against exact references when available. |
| `assemble_execution_manifest` | `data.pipeline` | Assemble source, schema, settings, code, dependency, seed, sampling, status and output metadata for the completed analysis run. |

</details>

<details>
<summary>Statistics (79)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `estimate_quantiles` | `statistics.descriptive` | Estimate requested quantiles with an explicit interpolation or approximation method and support counts. |
| `measure_concentration` | `statistics.descriptive` | Measure numerical or categorical concentration using a suitable declared measure such as Gini, HHI or entropy. |
| `estimate_empirical_distribution` | `statistics.distributions` | Return empirical cumulative probabilities and support as numerical values. |
| `compute_histogram` | `statistics.distributions` | Return bin edges, counts and optional weighted frequencies without rendering a plot. |
| `estimate_kernel_density` | `statistics.distributions` | Estimate density values on a supplied grid and report bandwidth and support limitations. |
| `detect_distribution_modes` | `statistics.distributions` | Identify evidence of one or multiple modes and report method and tuning sensitivity. |
| `fit_distribution_mixture` | `statistics.distributions` | Fit candidate mixture distributions and return component parameters, weights and fit diagnostics. |
| `analyze_distribution_boundaries` | `statistics.distributions` | Measure boundary mass, truncation indicators, floor or ceiling effects and values outside declared bounds. |
| `analyze_rounding_patterns` | `statistics.distributions` | Quantify rounding, heaping and quantization at specified numeric increments. |
| `fit_parametric_distribution` | `statistics.distributions` | Estimate parameters of a specified distribution and report support constraints and fitting diagnostics. |
| `compare_distribution_fits` | `statistics.distributions` | Compare compatible distribution fits using declared fit criteria and their uncertainty or limitations. |
| `test_normality` | `statistics.distributions` | Test normality using the selected procedure and report sample-size and assumption limitations. |
| `test_distribution_fit` | `statistics.distributions` | Run a goodness-of-fit test with suitable calibration, refitting estimated parameters within simulations when required. |
| `summarize_counts` | `statistics.counts` | Summarize count distributions, zeros and group support while checking count validity. |
| `estimate_event_rates` | `statistics.counts` | Estimate exposure-adjusted event rates and uncertainty by the requested groups. |
| `estimate_proportions` | `statistics.counts` | Estimate binomial or multinomial proportions with an explicitly selected interval method compatible with the declared outcome encoding. |
| `test_overdispersion` | `statistics.counts` | Test excess count variation relative to a specified reference model and its conditional mean structure. |
| `test_zero_inflation` | `statistics.counts` | Compare observed zeros with zeros expected under a specified count model. |
| `fit_diagnostic_count_model` | `statistics.counts` | Fit a diagnostic count model, including Poisson or negative binomial variants, and evaluate dispersion, zeros and residual behavior. |
| `extract_extreme_events` | `statistics.extremes` | Extract block maxima or peaks over threshold and report exceedance counts, exposure and declared declustering choices. |
| `fit_extreme_value_model` | `statistics.extremes` | Fit an extreme-value tail model such as GEV or generalized Pareto to extracted events. |
| `estimate_tail_risk` | `statistics.extremes` | Estimate exceedance probabilities or return levels with uncertainty and explicitly declared exposure units. |
| `analyze_extreme_threshold_sensitivity` | `statistics.extremes` | Compare tail parameters and risk estimates across candidate thresholds and report unstable or unsupported regions. |
| `compare_independent_means` | `statistics.comparisons` | Compare two independent-group means using conventional, Welch or robust trimmed-mean procedures. |
| `compare_independent_ranks` | `statistics.comparisons` | Compare two independent groups with Mann-Whitney or a specified rank procedure, recording ties and interpretation limits. |
| `compare_multiple_groups` | `statistics.comparisons` | Compare multiple independent groups using ANOVA, Welch ANOVA, Kruskal-Wallis or a declared permutation alternative. |
| `audit_repeated_measure_design` | `statistics.comparisons` | Check pairing, repeated-condition coverage, duplicate measurements and incomplete subjects before dependent comparisons. |
| `compare_paired_measurements` | `statistics.comparisons` | Compare two matched conditions using a paired t-test, Wilcoxon signed-rank or paired resampling. |
| `compare_repeated_measurements` | `statistics.comparisons` | Compare repeated conditions using repeated-measures ANOVA, Friedman or an appropriate subject-preserving alternative. |
| `build_contingency_table` | `statistics.comparisons` | Return observed and expected categorical cell counts, margins and sparse-cell warnings. |
| `test_categorical_independence` | `statistics.comparisons` | Test independence using chi-square, Fisher exact or a supported exact or resampling calibration for the table dimensions. |
| `compare_paired_categories` | `statistics.comparisons` | Compare paired categorical outcomes using McNemar or Cochran's Q with verified subject matching. |
| `test_stratified_categorical_association` | `statistics.comparisons` | Test categorical association across strata and assess effect homogeneity when supported by the selected procedure. |
| `compare_group_variances` | `statistics.comparisons` | Compare group variances using Levene, Fligner-Killeen or Bartlett and report assumptions. |
| `compare_covariance_matrices` | `statistics.comparisons` | Compare group covariance matrices using a declared parametric or calibrated resampling procedure. |
| `compare_multivariate_means` | `statistics.comparisons` | Compare multivariate group responses using Hotelling's T-squared, MANOVA or a specified compatible method. |
| `compare_multivariate_distances` | `statistics.comparisons` | Run a distance-based multivariate comparison with an appropriate exchangeability and resampling design. |
| `test_multivariate_dispersion` | `statistics.comparisons` | Test differences in within-group multivariate dispersion to support interpretation of distance-based comparisons. |
| `estimate_standardized_effect` | `statistics.effects` | Estimate a declared standardized effect such as Cohen's d or Hedges' g using the appropriate independent or paired design. |
| `estimate_rank_effect` | `statistics.effects` | Estimate a rank-based effect such as Cliff's delta or rank-biserial correlation with design-aware uncertainty. |
| `estimate_relative_risk` | `statistics.effects` | Estimate a risk ratio and uncertainty from explicitly oriented event and exposure counts. |
| `estimate_odds_ratio` | `statistics.effects` | Estimate an odds ratio, optionally across compatible strata, with declared cell orientation and small-count handling. |
| `test_equivalence` | `statistics.effects` | Test equivalence against explicit practical lower and upper margins using a compatible independent or paired procedure. |
| `estimate_correlation` | `statistics.association` | Estimate Pearson, Spearman, Kendall or point-biserial correlation with valid paired counts and supported uncertainty. |
| `estimate_categorical_association` | `statistics.association` | Estimate categorical association such as Cramer's V with contingency support and optional declared bias correction. |
| `estimate_correlation_ratio` | `statistics.association` | Measure numerical variation associated with category membership and report group support. |
| `estimate_ordinal_association` | `statistics.association` | Estimate ordinal association using declared category orders and tie handling. |
| `estimate_mutual_information` | `statistics.association` | Estimate mutual information for declared discrete or continuous variables and report estimator sensitivity and support. |
| `estimate_entropy` | `statistics.association` | Estimate discrete or differential entropy with declared units, assumptions and estimator settings. |
| `estimate_multivariate_information` | `statistics.association` | Estimate a specified multivariate information measure, such as total correlation, and record dimensionality limitations. |
| `estimate_nonlinear_dependence` | `statistics.association` | Estimate a nonlinear dependence score such as distance correlation or a kernel dependence statistic. |
| `test_nonlinear_independence` | `statistics.association` | Test nonlinear independence with a selected distance or kernel method and valid null calibration. |
| `estimate_partial_correlation` | `statistics.association` | Estimate correlation after accounting for explicitly selected conditioning variables. |
| `test_conditional_independence` | `statistics.association` | Test conditional independence using a method compatible with the declared data and conditioning structure. |
| `analyze_residualized_relationships` | `statistics.association` | Analyze relationships between residualized variables using an explicitly declared nuisance-fitting and evaluation design. |
| `fit_copula` | `statistics.association` | Fit a specified copula and marginals and return dependence parameters and fit diagnostics. |
| `estimate_tail_dependence` | `statistics.association` | Estimate upper or lower joint-tail dependence and report limited support at extreme quantiles. |
| `summarize_conditional_target` | `statistics.association` | Summarize target behavior within declared categories or numeric bins and return group support and uncertainty. |
| `estimate_interaction_effects` | `statistics.association` | Estimate specified feature interactions using a diagnostic model and valid evaluation. |
| `assess_subgroup_heterogeneity` | `statistics.association` | Compare a specified target statistic or association across subgroups with support and uncertainty. |
| `estimate_response_shape` | `statistics.association` | Estimate a numerical response shape using bins, smoothing or a valid model-based diagnostic; record extrapolation and confounding limits. |
| `detect_simpson_reversals` | `statistics.association` | Identify candidate reversals between aggregate and within-stratum associations with support and uncertainty. |
| `estimate_analytic_interval` | `statistics.inference` | Compute an analytic uncertainty interval using the declared reference distribution and any supported transformation. |
| `estimate_exact_interval` | `statistics.inference` | Compute a compatible exact interval for a specified count or proportion sampling model. |
| `bootstrap_statistic` | `statistics.inference` | Estimate a statistic's sampling distribution and interval using a bootstrap design that preserves declared dependence. |
| `jackknife_statistic` | `statistics.inference` | Estimate bias and uncertainty through declared leave-observation, leave-group or other supported jackknife deletions. |
| `run_permutation_test` | `statistics.inference` | Calibrate a test by exact or sampled permutations permitted under the declared null and exchangeability structure. |
| `run_monte_carlo_test` | `statistics.inference` | Calibrate a statistic against simulations from an explicitly specified null model and report simulation error. |
| `estimate_bayesian_interval` | `statistics.inference` | Estimate a credible interval from supplied posterior draws or a fully specified supported model, prior and observations. |
| `assess_resampling_precision` | `statistics.inference` | Assess Monte Carlo error and interval or p-value stability against declared precision targets. |
| `assess_rare_event_support` | `statistics.power` | Count rare events and non-events across requested groups and compare them with declared support requirements. |
| `assess_segment_sample_adequacy` | `statistics.power` | Assess segment-level observation, event and effective-sample support for explicitly selected analyses. |
| `estimate_statistical_power` | `statistics.power` | Estimate prospective power for an explicitly declared test, effect definition and independent or dependent study design. |
| `estimate_minimum_detectable_effect` | `statistics.power` | Estimate the smallest detectable effect for a declared design, significance level and desired power. |
| `estimate_required_sample_size` | `statistics.power` | Estimate required sample sizes for declared effect, allocation, variance and dependence assumptions. |
| `adjust_p_values` | `statistics.multiplicity` | Adjust p-values within explicitly supplied test families using a selected family-wise or false-discovery procedure. |
| `run_posthoc_comparisons` | `statistics.multiplicity` | Run compatible post-hoc group comparisons and retain raw or intrinsically adjusted results without applying duplicate corrections. |
| `annotate_adaptive_findings` | `statistics.multiplicity` | Label findings as prespecified, exploratory, adaptively selected or unknown using supplied analysis history. |
| `audit_multiple_testing` | `statistics.multiplicity` | Audit test-family membership, correction coverage, thresholds and adaptive-selection limitations. |

</details>

<details>
<summary>Features (15)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `find_duplicate_features` | `features.redundancy` | Find identical or tolerance-equivalent feature columns and return duplicate groups. |
| `analyze_matrix_conditioning` | `features.redundancy` | Compute matrix rank, singular values, eigenvalue summaries and condition numbers under declared scaling. |
| `estimate_variance_inflation` | `features.redundancy` | Estimate variance inflation factors and flag singular or unsupported feature combinations. |
| `estimate_covariance` | `features.redundancy` | Estimate ordinary, robust or regularized covariance with conditioning and sample-support diagnostics. |
| `find_redundant_feature_groups` | `features.redundancy` | Group strongly redundant features from a compatible association result using declared thresholds and grouping rules. |
| `decompose_numeric_structure` | `features.structure` | Fit PCA, kernel PCA, factor analysis, ICA or NMF and return component, loading and fit summaries. |
| `decompose_categorical_structure` | `features.structure` | Run correspondence or multiple-correspondence analysis and return category coordinates and inertia summaries. |
| `decompose_mixed_structure` | `features.structure` | Analyze mixed numerical and categorical factor structure using a suitable declared method. |
| `evaluate_structure_reconstruction` | `features.structure` | Measure reconstruction error, explained variation or a method-appropriate numerical fidelity measure. |
| `compute_pairwise_distances` | `features.structure` | Compute requested pairwise distances or their summaries under explicit scaling and output-size limits. |
| `analyze_neighborhood_structure` | `features.structure` | Summarize nearest-neighbor distances, connectivity, hubness and local support across selected neighborhood sizes. |
| `estimate_intrinsic_dimension` | `features.structure` | Estimate intrinsic dimensionality and report neighborhood, sampling and estimator sensitivity. |
| `measure_distance_concentration` | `features.structure` | Quantify how pairwise distances concentrate and how well nearest and farthest observations separate. |
| `compute_manifold_embedding` | `features.structure` | Compute a selected manifold embedding and return its numerical coordinates with the fit settings. |
| `evaluate_embedding_quality` | `features.structure` | Measure neighborhood preservation, trustworthiness, continuity or stress without rendering a plot. |

</details>

<details>
<summary>Patterns (17)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `cluster_numeric_data` | `patterns.clustering` | Cluster numeric observations using k-means, hierarchical, Gaussian-mixture, DBSCAN or HDBSCAN methods. |
| `cluster_categorical_data` | `patterns.clustering` | Cluster categorical observations using a compatible matching distance or categorical clustering algorithm. |
| `cluster_mixed_data` | `patterns.clustering` | Cluster mixed-type observations using declared scaling, categorical handling and a compatible algorithm. |
| `summarize_clusters` | `patterns.clustering` | Return cluster sizes, feature profiles, mixture membership summaries and noise or unassigned counts. |
| `evaluate_cluster_separation` | `patterns.clustering` | Compute silhouette, Davies-Bouldin or other requested cluster-separation measures with valid support. |
| `estimate_gap_statistic` | `patterns.clustering` | Compare within-cluster dispersion with a declared reference distribution across candidate cluster counts. |
| `assess_cluster_stability` | `patterns.clustering` | Assess cluster membership stability under resampling and optional parameter changes, accounting for label permutations and noise. |
| `compare_clusterings` | `patterns.clustering` | Measure agreement between aligned clusterings or between clusters and known segments using a selected compatible metric. |
| `detect_univariate_anomalies` | `patterns.anomalies` | Score univariate extremes using IQR or robust-score rules and return declared threshold exceedances. |
| `detect_multivariate_anomalies` | `patterns.anomalies` | Score multivariate anomalies using robust Mahalanobis distance, Isolation Forest or a compatible selected method. |
| `detect_local_anomalies` | `patterns.anomalies` | Score local anomalies using local outlier factors or nearest-neighbor distances with explicit threshold choices. |
| `detect_contextual_anomalies` | `patterns.anomalies` | Find unusual values relative to declared groups, time windows or other contextual conditions. |
| `compare_anomaly_results` | `patterns.anomalies` | Compare anomaly-score rankings and flagged-observation overlap across methods on aligned records. |
| `summarize_anomalies` | `patterns.anomalies` | Summarize anomaly counts, scores, affected groups and permitted examples without equating rare cases with data errors. |
| `mine_frequent_itemsets` | `patterns.rules` | Find frequently co-occurring categorical conditions with support counts and search-limit metadata. |
| `mine_association_rules` | `patterns.rules` | Generate association rules and return antecedents, consequents, support, confidence and lift. |
| `discover_unusual_subgroups` | `patterns.rules` | Search bounded combinations of categorical conditions for unusual target behavior and label findings as exploratory. |

</details>

<details>
<summary>Diagnostics (9)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `fit_diagnostic_model` | `diagnostics.models` | Fit a budget-limited linear, GLM, additive or tree diagnostic under the declared evaluation design and return its fit summary and diagnostics. |
| `evaluate_diagnostic_model` | `diagnostics.models` | Evaluate a diagnostic fit on permitted evaluation observations using declared metrics and dependence-aware uncertainty. |
| `analyze_model_residuals` | `diagnostics.models` | Analyze suitable out-of-sample residuals for bias, nonlinearity, variance patterns, dependence and influential observations. |
| `estimate_feature_importance` | `diagnostics.models` | Estimate model-specific or evaluation-based importance and document bias, correlation and evaluation limitations. |
| `estimate_conditional_feature_importance` | `diagnostics.models` | Estimate feature or feature-group importance while explicitly accounting for correlated predictors through a declared conditional method. |
| `assess_feature_selection_stability` | `diagnostics.models` | Measure selection frequencies and feature-set agreement across valid resamples without using a sealed holdout. |
| `run_sensitivity_analysis` | `diagnostics.sensitivity` | Re-run a selected analysis under declared sample, seed, threshold, time-window, missingness or transformation scenarios. |
| `analyze_influential_observations` | `diagnostics.sensitivity` | Measure how observations or declared groups influence a statistic, diagnostic model or conclusion. |
| `summarize_result_stability` | `diagnostics.sensitivity` | Summarize changes in estimates, uncertainty, rankings and conclusions across compatible sensitivity results. |

</details>

<details>
<summary>Time and sequences (32)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `validate_time_index` | `temporal.integrity` | Check ordering, valid timestamps, duplicate time keys and time-zone consistency within declared entities. |
| `analyze_observation_intervals` | `temporal.integrity` | Summarize inter-observation intervals, irregular spacing and interval changes by entity. |
| `find_time_gaps` | `temporal.integrity` | Find missing expected timestamps or extended gaps while respecting the supplied observation calendar. |
| `assess_temporal_coverage` | `temporal.integrity` | Measure observation coverage across the declared time range, periods and entities. |
| `assess_data_freshness` | `temporal.integrity` | Measure data age, ingestion lag and freshness violations relative to an explicit reference time. |
| `validate_observation_windows` | `temporal.integrity` | Check window ordering, duration, overlap and cross-field consistency against declared rules. |
| `estimate_time_trend` | `temporal.analysis` | Estimate a declared ordinary or robust temporal trend with uncertainty and sampling assumptions. |
| `decompose_time_series` | `temporal.analysis` | Return numerical trend, seasonal and residual components using STL or MSTL after validating sampling requirements. |
| `estimate_power_spectrum` | `temporal.analysis` | Estimate spectral power using an appropriate periodogram, spectral or Lomb-Scargle procedure for the observation schedule. |
| `measure_seasonality` | `temporal.analysis` | Quantify seasonal strength and repeatability at declared periods with coverage and detrending information. |
| `detect_periodicities` | `temporal.analysis` | Identify supported spectral peaks and candidate periods while recording resolution, aliasing and selection limitations. |
| `estimate_autocorrelation` | `temporal.analysis` | Estimate ACF or PACF at supported lags and report intervals and sampling assumptions. |
| `estimate_cross_correlation` | `temporal.analysis` | Estimate cross-correlation with an explicit lag-sign convention and temporal alignment. |
| `analyze_lagged_associations` | `temporal.analysis` | Measure lagged predictor-target associations without crossing declared entity boundaries. |
| `test_sequence_randomness` | `temporal.analysis` | Run a supported sequence-randomness or runs test with declared ordering and tie treatment. |
| `test_serial_correlation` | `temporal.analysis` | Test serial correlation, including Ljung-Box diagnostics with appropriate fitted-model degrees-of-freedom adjustments. |
| `test_stationarity` | `temporal.analysis` | Run ADF or KPSS with explicit deterministic terms and lag settings and state the selected test's null hypothesis. |
| `test_structural_breaks` | `temporal.analysis` | Test specified or unknown structural breaks using a suitable declared procedure, including CUSUM where applicable. |
| `detect_change_points` | `temporal.analysis` | Detect candidate changes in mean, variance or dependence under explicit segmentation and penalty rules. |
| `summarize_time_regimes` | `temporal.analysis` | Summarize observation support and distribution or dependence differences across detected time regimes. |
| `decompose_within_between_variation` | `temporal.longitudinal` | Separate within-group and between-group variation and report group sizes and imbalance. |
| `estimate_variance_components` | `temporal.longitudinal` | Estimate declared hierarchical variance components and record identifiability and boundary-fit limitations. |
| `fit_mixed_effects_diagnostic_model` | `temporal.longitudinal` | Fit a diagnostic mixed-effects model with declared group structure and valid evaluation. |
| `fit_gee_diagnostic_model` | `temporal.longitudinal` | Fit a diagnostic generalized estimating equation model with declared clusters, link family and working dependence. |
| `check_longitudinal_model_assumptions` | `temporal.longitudinal` | Check longitudinal residual patterns, covariance choices, convergence and group-level support for a fitted diagnostic model. |
| `analyze_interarrival_times` | `temporal.events` | Summarize time between successive events within entities and optional event types. |
| `analyze_state_dwell_times` | `temporal.events` | Measure durations spent in states and distinguish completed dwell periods from censored final states. |
| `estimate_state_transitions` | `temporal.events` | Return state-transition counts and probabilities at a declared event lag. |
| `validate_event_sequences` | `temporal.events` | Find invalid event ordering, missing prerequisites and prohibited transitions under supplied sequence rules. |
| `mine_event_sequences` | `temporal.events` | Find recurring bounded event sequences and return occurrence support and search constraints. |
| `analyze_cohort_progression` | `temporal.events` | Summarize cohort progression through declared stages and relative time periods. |
| `estimate_retention` | `temporal.events` | Estimate explicitly defined cohort retention while accounting for cohort maturity and observation coverage. |

</details>

<details>
<summary>Specialized data (39)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `profile_followup` | `specialized.survival` | Summarize follow-up duration, events, observations at risk and coverage by group. |
| `audit_censoring_and_truncation` | `specialized.survival` | Check censoring and entry-time encoding and quantify observed censoring or truncation patterns without proving non-informative censoring. |
| `estimate_survival_curve` | `specialized.survival` | Estimate supported survival probabilities, intervals and at-risk counts at requested times. |
| `estimate_cumulative_hazard` | `specialized.survival` | Estimate cumulative hazards and uncertainty using a method compatible with the supplied survival design. |
| `compare_survival_groups` | `specialized.survival` | Compare group survival using a declared log-rank or compatible stratified procedure. |
| `fit_survival_diagnostic_model` | `specialized.survival` | Fit and evaluate a suitable diagnostic survival model while preserving censoring and entry-time semantics. |
| `check_survival_model_assumptions` | `specialized.survival` | Check supported model assumptions such as proportional hazards and report residual, convergence and follow-up limitations. |
| `validate_spatial_coordinates` | `specialized.spatial` | Check coordinate ranges, missing values, declared reference systems and expected spatial bounds. |
| `summarize_spatial_coverage` | `specialized.spatial` | Summarize spatial extent, occupied regions, density and coverage gaps using declared region or grid definitions. |
| `compute_spatial_distances` | `specialized.spatial` | Compute projected or geodesic distances with explicit units and bounded numerical output. |
| `build_spatial_weights` | `specialized.spatial` | Construct a reproducible spatial-neighbor or distance-weight definition and report isolated observations. |
| `test_global_spatial_autocorrelation` | `specialized.spatial` | Estimate and test global Moran's I or Geary's C under the declared spatial weights and null model. |
| `test_local_spatial_autocorrelation` | `specialized.spatial` | Return local spatial-association statistics and raw test results for subsequent multiple-testing control. |
| `detect_spatial_hotspots` | `specialized.spatial` | Identify candidate high- or low-value spatial concentrations using a declared hotspot statistic and calibrated uncertainty. |
| `assess_spatial_role_dependence` | `specialized.spatial` | Measure spatial proximity and shared entities across dataset roles that may compromise spatially independent evaluation. |
| `validate_composition` | `specialized.compositional` | Check non-negativity, sum constraints, missing components and zero counts for compositional observations. |
| `analyze_log_ratios` | `specialized.compositional` | Return declared CLR, ALR or ILR log-ratio summaries with zero treatment and basis or reference metadata. |
| `compute_compositional_distances` | `specialized.compositional` | Compute Aitchison distances or their summaries after explicit compositional zero handling. |
| `analyze_compositional_zero_sensitivity` | `specialized.compositional` | Compare selected compositional findings across declared zero-replacement or zero-exclusion scenarios. |
| `validate_cyclic_measurements` | `specialized.directional` | Check declared cyclic units, periods, ranges and invalid or ambiguous angle encodings. |
| `summarize_circular_data` | `specialized.directional` | Estimate circular mean, dispersion and concentration and explain undefined summaries when directions cancel. |
| `summarize_directional_data` | `specialized.directional` | Summarize normalized multidimensional directions, mean direction and directional concentration under declared coordinates. |
| `test_directional_structure` | `specialized.directional` | Test a specified circular or directional structure, such as uniformity or concentration around a supplied direction. |
| `build_entity_graph` | `specialized.graphs` | Build a declared entity graph and record node identity, edge aggregation, direction, weights and isolated-node handling. |
| `summarize_graph_structure` | `specialized.graphs` | Summarize node and edge counts, degree distributions, density, self-loops and connectedness. |
| `find_connected_components` | `specialized.graphs` | Identify connected, weakly connected or strongly connected components as appropriate to the graph. |
| `estimate_graph_centrality` | `specialized.graphs` | Estimate selected centrality measures and document weighting, disconnectedness and approximation choices. |
| `detect_graph_communities` | `specialized.graphs` | Identify communities using a method compatible with the graph and return membership, support and quality summaries. |
| `estimate_graph_assortativity` | `specialized.graphs` | Measure degree or attribute assortativity under the declared graph and edge-weight semantics. |
| `analyze_relational_dependence` | `specialized.graphs` | Measure attribute dependence across related entities and optional graph connections between evaluation roles. |
| `validate_functional_observations` | `specialized.functional` | Check curve identities, domains, sampling grids, coverage, missing values and declared alignment requirements. |
| `summarize_functional_data` | `specialized.functional` | Return numerical mean or robust curves, variation summaries and support across a declared functional domain. |
| `align_functional_data` | `specialized.functional` | Align repeated curves under explicit landmark or registration constraints and record the transformations. |
| `decompose_functional_variation` | `specialized.functional` | Estimate functional principal components or another selected functional decomposition and summarize explained variation. |
| `compare_functional_shapes` | `specialized.functional` | Compare group curve shapes or trajectories with a compatible functional statistic and dependence-aware uncertainty. |
| `compute_persistent_homology` | `specialized.topology` | Compute persistent topological features from a point cloud, distance matrix or compatible supplied complex. |
| `summarize_topological_features` | `specialized.topology` | Return requested numerical persistence summaries, including connectivity, loops and higher-dimensional feature lifetimes. |
| `compare_topological_signatures` | `specialized.topology` | Measure differences between compatible persistence diagrams or topological summaries. |
| `assess_topological_stability` | `specialized.topology` | Assess how topological features change under declared sampling, perturbation or filtration scenarios. |

</details>

<details>
<summary>Evaluation (23)</summary>

| Function | Module | What it does |
| --- | --- | --- |
| `compare_univariate_distributions` | `evaluation.shift` | Compare selected marginal distributions using compatible numerical distances or two-sample tests and report support. |
| `compare_multivariate_distributions` | `evaluation.shift` | Measure joint distribution differences using a declared distance, energy or kernel two-sample procedure. |
| `fit_adversarial_validation_model` | `evaluation.shift` | Evaluate how well a diagnostic classifier distinguishes dataset membership under leakage-safe evaluation. |
| `localize_distribution_shift` | `evaluation.shift` | Locate observed shift by feature, group or time period without inferring a causal shift mechanism from marginal differences alone. |
| `profile_sensitive_groups` | `evaluation.fairness` | Summarize permitted sensitive-group definitions, support, missing membership and weighted representation. |
| `assess_intersectional_coverage` | `evaluation.fairness` | Measure intersectional group support and identify sparse or absent intersections within explicit search limits. |
| `compare_group_label_distributions` | `evaluation.fairness` | Compare observed label rates or target distributions across groups with uncertainty and small-group limitations. |
| `compare_group_data_quality` | `evaluation.fairness` | Compare missingness, validity and other specified data-quality indicators across groups. |
| `measure_proxy_associations` | `evaluation.fairness` | Measure associations between permitted sensitive attributes and potential proxy features without treating association as proof of unfairness. |
| `estimate_selection_propensity` | `evaluation.confounding` | Estimate observed inclusion propensity from measured covariates and document unavailable selection causes. |
| `assess_covariate_balance` | `evaluation.confounding` | Compare covariate balance between declared groups using standardized or distribution-based diagnostics. |
| `assess_population_overlap` | `evaluation.confounding` | Assess observed common support and near-positivity violations under a declared overlap diagnostic. |
| `evaluate_negative_controls` | `evaluation.confounding` | Evaluate supplied negative-control relationships and record assumptions needed to interpret their results. |
| `check_supplied_causal_assumptions` | `evaluation.confounding` | Evaluate testable implications of supplied causal assumptions and explicitly list assumptions that observed data cannot verify. |
| `analyze_confounding_sensitivity` | `evaluation.confounding` | Quantify sensitivity of a supplied estimand to explicitly parameterized unmeasured confounding assumptions. |
| `audit_prediction_time_availability` | `evaluation.leakage` | Check whether selected feature values were available at prediction time using supplied availability, update and lineage evidence. |
| `detect_target_derived_features` | `evaluation.leakage` | Identify direct or indirect target-derived features from explicit lineage and transformation evidence. |
| `detect_suspicious_target_proxies` | `evaluation.leakage` | Flag unusually predictive or target-like features for investigation while keeping statistical suspicion separate from confirmed leakage. |
| `measure_role_overlap` | `evaluation.leakage` | Measure exact-record, entity and declared near-duplicate overlap between permitted dataset roles. |
| `audit_temporal_role_boundaries` | `evaluation.leakage` | Check chronological separation, purge or embargo rules and overlapping label or feature windows across dataset roles. |
| `audit_preprocessing_fit_scope` | `evaluation.leakage` | Check which data roles fitted preprocessing, selection and other learned transformations using recorded execution evidence. |
| `audit_holdout_access` | `evaluation.leakage` | Audit recorded final-holdout reads and uses against the declared holdout policy and mark incomplete access evidence. |
| `evaluate_leakage_evidence` | `evaluation.leakage` | Combine leakage findings into confirmed violations, supported suspicions and unevaluated checks without converting missing evidence into a pass. |

</details>
