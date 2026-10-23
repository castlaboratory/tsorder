# Q5 — CPU cost per series (median), log-log in n.

source("benchmark/analysis/common.R")

tim <- read_csv(file.path(results_dir, "q5_time.csv"),
                col_types = cols(run = col_character(), dgp = col_character(),
                                 n = col_integer(), estimator = col_character(),
                                 median_us = col_double()))

p <- tim |>
  ggplot(aes(n, median_us, colour = estimator)) +
  geom_line() + geom_point(size = 1) +
  scale_x_log10() + scale_y_log10() +
  labs(title = "Q5 - median CPU time per series",
       x = "n (log)", y = "microseconds (log)", colour = NULL) +
  theme_bench()
save_fig("fig-q5-cost", p, height = 5.5)
