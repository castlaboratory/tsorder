test_that("selectors recover known orders", {
  set.seed(11)
  ar2 <- as.numeric(arima.sim(list(ar = c(1.2, -0.6)), 2000))
  ma2 <- as.numeric(arima.sim(list(ma = c(0.6, 0.3)), 4000))
  wn <- rnorm(2000)

  expect_identical(fh_order(ar2)$order, 2L)
  expect_identical(ic_order(ar2, "bic")$order, 2L)
  expect_identical(fh_order(ma2, statistic = "acf_bartlett")$order, 2L)
  expect_identical(fh_order(ma2, statistic = "inverse_pacf")$order, 2L)
  expect_identical(fh_order(wn)$order, 0L)
  expect_identical(svc_order(ar2)$order, 2L)
  expect_identical(hankel_rank_order(ar2)$order, 2L)
  expect_lte(hankel_svt_order(wn)$order, 0L)

  rw <- cumsum(rnorm(1500))
  expect_identical(integration_order(rw)$d, 1L)
  expect_identical(integration_order(rnorm(1500))$d, 0L)
})

test_that("order_select returns a well-formed fit for every family", {
  set.seed(5)
  x <- as.numeric(arima.sim(list(ar = c(1.2, -0.6)), 1000))
  for (m in c("pacf", "bic", "jirak", "svc", "hankel_rank")) {
    fit <- order_select(x, method = m)
    expect_s3_class(fit, "tsorder_fit")
    expect_true(is.numeric(fit$order) || is.integer(fit$order))
    expect_s3_class(fit$path, "tbl_df")
    expect_s3_class(ggplot2::autoplot(fit), "ggplot")
  }
  expect_message(print(order_select(x)), "Selected")
})

test_that("errors surface as R conditions, not crashes", {
  expect_error(fh_order(c(1, 2, 3)), "short")
  expect_error(ic_order(rnorm(100), criterion = "nope"), "criterion")
})
