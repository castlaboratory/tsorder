# Q7 — memory rule for the PACF-clip: poly-log vs n^0.3 vs n^0.4 vs 2 sqrt(n).
# Decides the Memory default (T-2, D-06).

source("benchmark/analysis/common.R")

orders <- read_orders("q7")
acc <- accuracy(orders)

p1 <- acc |>
  filter(dgp != "wn") |>
  ggplot(aes(n, p_correct, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp) +
  scale_x_n() +
  labs(title = "Q7 - memory rule: P(correct) for the PACF-clip (bic c = 1.5)",
       x = "n (log)", y = "P(correct)", colour = NULL) +
  theme_bench()
save_fig("fig-q7-accuracy", p1)

p2 <- level(orders) |>
  filter(dgp == "wn") |>
  ggplot(aes(n, p_over, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  scale_x_n() +
  labs(title = "Q7 - overestimation on white noise by memory rule",
       x = "n (log)", y = "P(order > 0)", colour = NULL) +
  theme_bench()
save_fig("fig-q7-level", p2, height = 5)
