test_that("fit_arima closes the selection-to-fit loop", {
  set.seed(2)
  x <- arima.sim(list(ar = c(1.2, -0.6)), 600)
  fit <- fit_arima(x)
  expect_s3_class(fit, "Arima")
  expect_identical(attr(fit, "tsorder_d"), 0L)
  expect_identical(names(coef(fit))[1:2], c("ar1", "ar2"))
  expect_length(predict(fit, n.ahead = 4)$pred, 4)

  rw <- cumsum(as.numeric(arima.sim(list(ar = 0.5), 800)))
  fit1 <- fit_arima(rw, method = "bic")
  expect_identical(attr(fit1, "tsorder_d"), 1L)

  y <- arima.sim(list(ar = 0.5, ma = 0.8), 800)
  fh <- fit_arima(y, method = "hr")
  expect_identical(attr(fh, "tsorder")$p, 1L)

  expect_error(fit_arima(x, method = "svc"), "McMillan")
})

test_that("the fitted object works with forecast and ts inputs pass through", {
  skip_if_not_installed("forecast")
  x <- ts(as.numeric(arima.sim(list(ar = c(1.2, -0.6)), 400)), frequency = 12)
  fit <- fit_arima(x)
  fc <- forecast::forecast(fit, h = 6)
  expect_length(fc$mean, 6)
  expect_identical(stats::frequency(fc$mean), 12)
})

test_that("tidy and glance follow broom conventions", {
  set.seed(3)
  x <- as.numeric(arima.sim(list(ar = c(1.2, -0.6)), 500))
  fit <- order_select(x)
  td <- generics::tidy(fit)
  expect_s3_class(td, "tbl_df")
  expect_true(all(c("lag", "studentized", "kept") %in% names(td)))
  gl <- generics::glance(fit)
  expect_identical(nrow(gl), 1L)
  expect_identical(gl$order, as.integer(fit$order))
  gl2 <- generics::glance(order_select(x, method = "bic"))
  expect_true(is.na(gl2$tau))
})

test_that("as_tibble() returns the tidy decision path", {
  set.seed(5)
  x <- as.numeric(arima.sim(list(ar = c(1.2, -0.6)), 600))
  fit <- order_select(x)
  expect_identical(tibble::as_tibble(fit), generics::tidy(fit))
  expect_true(all(c("lag", "studentized", "kept") %in%
                    names(tibble::as_tibble(fit))))
})
