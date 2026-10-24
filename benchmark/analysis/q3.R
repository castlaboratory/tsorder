# Q3 — contamination: robust PACF vs classical selectors.

source("benchmark/analysis/common.R")

acc <- accuracy(read_orders("q3")) |>
  mutate(dgp = factor(dgp, levels = c("ar2_clean", "ar2_outliers", "ar2_t3", "ar2_t2")))

p <- acc |>
  ggplot(aes(n, p_correct, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp, nrow = 1) +
  scale_x_n() +
  labs(title = "Q3 - AR(2) under contamination: P(correct)",
       subtitle = "clean | 5% additive N(0,25) outliers | t3 innovations | t2 innovations",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q3-contamination", p, height = 5)
