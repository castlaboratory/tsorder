simulate_ar2 <- function(n, seed) {
  set.seed(seed)
  as.numeric(arima.sim(list(ar = c(1.2, -0.6)), n))
}

test_that("Levinson PACF and AIC order match stats::ar.yw", {
  x <- simulate_ar2(800, 1)
  k <- 20
  ours <- rs_levinson(x, as.integer(k))
  theirs <- stats::ar.yw(x, aic = FALSE, order.max = k, demean = TRUE)
  expect_equal(ours$pacf, as.numeric(theirs$partialacf), tolerance = 1e-10)
  for (seed in 1:5) {
    y <- simulate_ar2(600, seed)
    expect_identical(ic_order(y, "aic")$order,
                     as.integer(stats::ar.yw(y)$order))
  }
})

test_that("qn matches robustbase::Qn to near machine precision", {
  skip_if_not_installed("robustbase")
  set.seed(42)
  for (n in c(10, 57, 200)) {
    z <- rnorm(n)
    expect_equal(qn(z), robustbase::Qn(z), tolerance = 1e-12)
  }
})

test_that("eacf table matches TSA::eacf", {
  skip_if_not_installed("TSA")
  x <- simulate_ar2(500, 7)
  ours <- eacf(x, max_ar = 7, max_ma = 7)
  theirs <- suppressWarnings(TSA::eacf(x, ar.max = 7, ma.max = 7))
  expect_equal(unname(ours$table), unname(theirs$eacf), tolerance = 1e-10)
})

test_that("hannan_rissanen is sane against forecast::auto.arima on ARMA(1,1)", {
  skip_if_not_installed("forecast")
  set.seed(3)
  x <- as.numeric(arima.sim(list(ar = 0.5, ma = 0.8), 1000))
  hr <- hannan_rissanen(x)
  expect_identical(c(hr$p, hr$q), c(1L, 1L))
  # auto.arima is a reference point, not ground truth (its stepwise search
  # may land on a neighbouring model); require only a low-order ARMA.
  aa <- forecast::auto.arima(x, d = 0, seasonal = FALSE, stepwise = TRUE)
  ord <- unname(forecast::arimaorder(aa)[c(1, 3)])
  expect_true(all(ord <= 2) && max(ord) >= 1)
})
