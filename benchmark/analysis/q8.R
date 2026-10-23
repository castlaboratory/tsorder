# Q8 — ARMA (p, q): EACF vertex vs Hannan-Rissanen (orders encoded 10p + q;
# -1 = estimator error, which for eacf means no clean vertex).

source("benchmark/analysis/common.R")

orders <- read_orders("q8")

acc <- orders |>
  group_by(dgp, n, estimator) |>
  summarise(
    reps = sum(count),
    p_correct = sum(count[order == true_order]) / reps,
    p_none = sum(count[order < 0]) / reps,
    .groups = "drop"
  )

p <- acc |>
  ggplot(aes(n, p_correct, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q8 - P((p,q) exactly right)",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q8-pq", p, height = 6.5)
