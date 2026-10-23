# Q1 — PACF-clip vs information criteria vs JirakBand, by n and DGP.
# Figures: fig-q1-accuracy (P(correct)), fig-q1-level (overestimation on
# white noise = level under H0), fig-q1-sensitivity (c in {1, 1.5, 2}).

source("benchmark/analysis/common.R")

orders <- read_orders("q1")
acc <- accuracy(orders)

main <- c("bic_yw", "hq_yw", "aic_yw", "pacf_bic15", "jirak_gumbel05", "jirak_gumbel_invn")

p1 <- acc |>
  filter(estimator %in% main, dgp != "wn") |>
  ggplot(aes(n, p_correct, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q1 - P(selected order = true order)",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q1-accuracy", p1)

p2 <- level(orders) |>
  filter(dgp == "wn") |>
  ggplot(aes(n, p_over, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  geom_hline(yintercept = 0.05, linetype = 2) +
  scale_x_n() +
  labs(title = "Q1 - overestimation on white noise (level under H0)",
       x = "n (log)", y = "P(order > 0)", colour = NULL) +
  theme_bench()
save_fig("fig-q1-level", p2, height = 5)

p3 <- acc |>
  filter(grepl("^pacf_bic", estimator)) |>
  ggplot(aes(n, p_correct, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q1 - sensitivity of the PACF-clip to c (threshold sqrt(c log n))",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q1-sensitivity", p3)
