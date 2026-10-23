# Q9 — observation noise: who degrades least. The observed process is
# ARMA(2,2) with McMillan degree 2, so svc targets 2 structurally; AR-order
# selectors are misspecified under noise. hr is scored on (p,q) = (2,2)
# for the noisy DGPs ((2,0) when clean).

source("benchmark/analysis/common.R")

orders <- read_orders("q9")

acc <- orders |>
  mutate(target = case_when(
    estimator == "hr" & dgp == "snr_inf" ~ 20L,
    estimator == "hr" ~ 22L,
    TRUE ~ 2L
  )) |>
  group_by(dgp, n, estimator) |>
  summarise(p_correct = sum(count[order == target]) / sum(count), .groups = "drop") |>
  mutate(dgp = factor(dgp, levels = c("snr_inf", "snr_10", "snr_3", "snr_1")))

p <- acc |>
  ggplot(aes(n, p_correct, colour = estimator,
             linewidth = estimator == "svc_logt_d1")) +
  geom_line() + geom_point(size = 1) +
  scale_linewidth_manual(values = c(`FALSE` = 0.4, `TRUE` = 1.1), guide = "none") +
  facet_wrap(~dgp, nrow = 1) +
  scale_x_n() +
  labs(title = "Q9 - AR(2) + observation noise: P(correct target)",
       subtitle = "AR selectors target order 2 (misspecified under noise); svc targets the McMillan degree 2 (bold); hr targets (2,2) under noise",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q9-obsnoise", p, height = 5)
