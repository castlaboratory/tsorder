# Q2 — ensemble vs best member vs BIC. Decides whether the ensemble stays.

source("benchmark/analysis/common.R")

acc <- accuracy(read_orders("q2"))

p <- acc |>
  ggplot(aes(n, p_correct, colour = estimator,
             linewidth = grepl("^ens_", estimator))) +
  geom_line() + geom_point(size = 1) +
  scale_linewidth_manual(values = c(`FALSE` = 0.4, `TRUE` = 1.1), guide = "none") +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q2 - ensembles (bold) vs members vs BIC",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q2-ensemble", p)

# Numeric table: is the ensemble ever better than its best member?
tab <- acc |>
  mutate(group = case_when(
    grepl("^ens_", estimator) ~ "ensemble",
    estimator == "bic_yw" ~ "bic",
    TRUE ~ "member"
  )) |>
  group_by(dgp, n, group) |>
  summarise(best = max(p_correct), .groups = "drop") |>
  pivot_wider(names_from = group, values_from = best) |>
  mutate(ens_minus_best_member = ensemble - member)
write_csv(tab, file.path(results_dir, "q2_summary.csv"))
print(tab |> group_by(dgp) |> summarise(max_gain = max(ens_minus_best_member)), n = 50)
