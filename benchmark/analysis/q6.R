# Q6 — infinite order: one-step-ahead RMSE of the selected model,
# relative to AIC (Yang 2005's efficiency/consistency trade-off in numbers).

source("benchmark/analysis/common.R")

pred <- read_csv(file.path(results_dir, "q6_pred.csv"),
                 col_types = cols(run = col_character(), dgp = col_character(),
                                  n = col_integer(), estimator = col_character(),
                                  sum_sq_err = col_double(),
                                  sum_sq_err_sq = col_double(),
                                  count = col_integer())) |>
  mutate(mse = sum_sq_err / count)

rel <- pred |>
  left_join(pred |> filter(estimator == "aic_yw") |>
              select(dgp, n, mse_aic = mse), by = c("dgp", "n")) |>
  mutate(rel_rmse = sqrt(mse / mse_aic))

p <- rel |>
  filter(estimator != "aic_yw") |>
  ggplot(aes(n, rel_rmse, colour = estimator)) +
  geom_hline(yintercept = 1, linetype = 2) +
  geom_line() + geom_point(size = 1) +
  facet_wrap(~dgp, scales = "free_y") +
  scale_x_n() +
  labs(title = "Q6 - one-step RMSE relative to AIC (AIC = 1)",
       subtitle = "ma1_inf and arfima03 are AR(infinity); ar2_strong is the finite-order control",
       x = "n (log)", y = "RMSE / RMSE(AIC)", colour = NULL) +
  theme_bench()
save_fig("fig-q6-prediction", p, height = 5)
