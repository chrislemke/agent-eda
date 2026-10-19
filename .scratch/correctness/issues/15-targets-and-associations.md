# 15: Targets and associations

What to build:
- **`compute_association_matrix`:** a bias-adjusted correlation ratio by default for mixed pairs; its own reason when each row is its own level; a pair budget that accounts for rank methods; pair values that don't depend on the other columns.
- **`profile_target`:** findings for missing declared classes, a regression task on a two-value target, and labels that differ only by case or spaces; `expected_labels` working with datetime and period categories.

Read these sections of `.scratch/correctness/spec.md`: Judging ordinary data (`compute_association_matrix` and `profile_target`), and test group 10. This ticket covers user stories 9 and 10. Findings: AN2-5, AN2-6, AX-14, AX-16, STD-13, STD-44, AN2-12, AN2 questionable decision 3 (rank-method cost), SPEC questionable decision (trivial perfect associations).

Blocked by: 01

Status: ready-for-agent

- [ ] The default mixed-pair method is a bias-adjusted correlation ratio: the square root of epsilon squared, clipped at 0. The plain correlation ratio stays available as a method.
- [ ] On 20 runs of 60 rows, a 40-level noise column against a noise column scores near 0 under the default, not 0.74. The README's example with `email` no longer prints perfect associations.
- [ ] The bias-adjusted ratio matches a hand-computed epsilon squared within 1e-12.
- [ ] A pair left null because each row is its own level gets its own reason and wording.
- [ ] The pair budget accounts for the cost of rank methods. On 100 numeric columns of 1,000,000 rows with `numeric_method="kendall"`, the call returns `budget_exceeded` with a reason, instead of running for minutes.
- [ ] A pair's value doesn't change with the other columns in the call, checked to the last digit. Or, if that can't hold, the docs say so and the Comments record why.
- [ ] The `budget_exceeded` footer doesn't offer a matrix when no pair was computed.
- [ ] `profile_target` with `expected_labels=["setosa", "versicolor", "virginica"]`, on data holding only the first two, reports a violation `missing_expected_classes`. "virginica" appears at count 0 in the class table and counts in the minority share and the imbalance ratio.
- [ ] `profile_target`'s two new suspected findings:
  - `task="regression"` on a 0/1 column;
  - multiclass labels `"Yes", "yes", " yes", "NO", "No"`, flagged as labels that differ only by case or surrounding spaces.
- [ ] `expected_labels` works with datetime and period category targets, and its error no longer speaks of sentinel values (STD-13).
- [ ] The new finding codes are declared, documented and listed in the README.
- [ ] The finish checks pass.

## Comments
