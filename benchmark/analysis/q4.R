# Q4 — McMillan degree: accuracy by n and DGP, level on white noise,
# and the MbC verdict (D-26).

source("benchmark/analysis/common.R")

orders <- read_orders("q4")
acc <- accuracy(orders)

p1 <- acc |>
  filter(dgp != "wn") |>
  ggplot(aes(n, p_correct, colour = estimator,
             linewidth = estimator == "mbc")) +
  geom_line() + geom_point(size = 1) +
  scale_linewidth_manual(values = c(`FALSE` = 0.4, `TRUE` = 1.1), guide = "none") +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q4 - McMillan degree: P(correct) by route (MbC bold)",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q4-accuracy", p1, height = 7)

p2 <- level(orders) |>
  filter(dgp == "wn") |>
  ggplot(aes(n, p_over, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  geom_hline(yintercept = 0.05, linetype = 2) +
  scale_x_n() +
  labs(title = "Q4 - overestimation on white noise (level under H0)",
       x = "n (log)", y = "P(degree > 0)", colour = NULL) +
  theme_bench()
save_fig("fig-q4-level", p2, height = 5)

# MbC verdict table: is the mode ever better than its best voter?
voters <- c("svc_logt_d2", "cc_bic", "rank_logdecay", "svt_c1")
tab <- acc |>
  mutate(group = case_when(
    estimator == "mbc" ~ "mbc",
    estimator %in% voters ~ "voter",
    TRUE ~ "other"
  )) |>
  filter(group != "other") |>
  group_by(dgp, n, group) |>
  summarise(best = max(p_correct), worst = min(p_correct), .groups = "drop") |>
  pivot_wider(names_from = group, values_from = c(best, worst)) |>
  mutate(mbc_minus_best_voter = best_mbc - best_voter,
         mbc_minus_worst_voter = best_mbc - worst_voter)
write_csv(tab, file.path(results_dir, "q4_mbc_summary.csv"))
print(tab |> group_by(dgp) |>
        summarise(max_gain_vs_best = max(mbc_minus_best_voter),
                  min_gain_vs_worst = min(mbc_minus_worst_voter)), n = 20)
