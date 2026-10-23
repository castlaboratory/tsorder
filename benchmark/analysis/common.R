# Shared readers and theme for the benchmark figures.
# Run any qX.R from the repository root: Rscript benchmark/analysis/q1.R

library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)

results_dir <- "benchmark/results"
figures_dir <- "benchmark/figures"
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

read_orders <- function(q) {
  read_csv(
    file.path(results_dir, paste0(q, "_orders.csv")),
    col_types = cols(
      run = col_character(), dgp = col_character(), n = col_integer(),
      estimator = col_character(), order = col_integer(),
      count = col_integer(), true_order = col_integer()
    )
  )
}

# P(correct), P(under), P(over) per (dgp, n, estimator); order -1 = failure.
accuracy <- function(orders) {
  orders |>
    group_by(dgp, n, estimator) |>
    summarise(
      reps = sum(count),
      p_correct = sum(count[order == true_order]) / reps,
      p_under = sum(count[order >= 0 & order < true_order]) / reps,
      p_over = sum(count[order > true_order]) / reps,
      p_fail = sum(count[order < 0]) / reps,
      .groups = "drop"
    )
}

# Overestimation level on a true-order DGP (for wn this is the H0 level).
level <- function(orders) {
  orders |>
    group_by(dgp, n, estimator) |>
    summarise(
      p_over = sum(count[order > true_order]) / sum(count),
      .groups = "drop"
    )
}

# Okabe-Ito: colourblind-safe categorical palette, fixed assignment order.
okabe_ito <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9",
               "#E69F00", "#999999", "#000000", "#F0E442")
options(ggplot2.discrete.colour = list(okabe_ito),
        ggplot2.discrete.fill = list(okabe_ito))

theme_bench <- function() {
  theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold")
    )
}

scale_x_n <- function() {
  scale_x_log10(breaks = c(50, 200, 800, 3200, 10000),
                labels = c("50", "200", "800", "3.2k", "10k"))
}

save_fig <- function(name, plot, width = 10, height = 6.5) {
  path <- file.path(figures_dir, paste0(name, ".pdf"))
  ggsave(path, plot, width = width, height = height)
  ggsave(sub("[.]pdf$", ".png", path), plot, width = width, height = height, dpi = 150)
  message("wrote ", path)
}
