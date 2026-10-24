#' Select orders and fit the model
#'
#' Closes the loop from selection to a fitted model: estimates the order of
#' integration (unless `d` is given), runs the chosen order selector on the
#' suitably differenced series, and fits the resulting ARIMA specification
#' with [stats::arima()]. The returned object is a standard `"Arima"` fit,
#' so the usual ecosystem applies directly: `predict()`,
#' `forecast::forecast()`, `broom::tidy()`, residual diagnostics, etc.
#'
#' @inheritParams order_select
#' @param method As in [order_select()] (AR-targeted statistics and criteria
#'   give `ARIMA(p, d, 0)`; MA-targeted statistics give `ARIMA(0, d, q)`),
#'   plus `"hr"` for Hannan-Rissanen, which selects `p` and `q` jointly.
#'   McMillan-degree methods do not define `(p, q)` and are not accepted.
#' @param d Order of integration; `NULL` (default) estimates it with
#'   [integration_order()].
#' @param include.mean Passed to [stats::arima()] (ignored when `d > 0`,
#'   as there).
#' @param ... Passed to the underlying selector.
#' @return A [stats::arima()] fit with two extra attributes: `tsorder`
#'   (the selection object) and `tsorder_d` (the integration order used).
#' @examples
#' set.seed(1)
#' x <- arima.sim(list(ar = c(1.2, -0.6)), 300)
#' fit <- fit_arima(x)
#' fit
#' predict(fit, n.ahead = 6)$pred
#' @export
fit_arima <- function(x, method = "pacf", d = NULL, include.mean = TRUE, ...) {
  if (is.null(d)) d <- integration_order(as.double(x))$d
  xd <- if (d > 0) diff(as.double(x), differences = d) else as.double(x)
  ma_methods <- c("acf_bartlett", "inverse_pacf", "acf", "acf_robust")
  if (identical(method, "hr")) {
    sel <- hannan_rissanen(xd, ...)
    ord <- c(sel$p, d, sel$q)
  } else {
    sel <- order_select(xd, method = method, ...)
    if (identical(sel$family, "mcmillan")) {
      cli::cli_abort(c(
        "{.val {method}} estimates the McMillan degree, not an ARMA order.",
        i = "Use an AR/MA statistic, a criterion, or {.val hr} with {.fn fit_arima}."
      ))
    }
    ord <- if (method %in% ma_methods) c(0L, d, sel$order) else c(sel$order, d, 0L)
  }
  fit <- stats::arima(x, order = ord, include.mean = include.mean)
  attr(fit, "tsorder") <- sel
  attr(fit, "tsorder_d") <- d
  fit
}

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy the decision path of a tsorder fit
#'
#' Returns the decision path as a tibble, one row per candidate (lag or
#' order), following the broom convention so tsorder plugs into tidy
#' workflows.
#'
#' @param x A [order_select()] result.
#' @param ... Unused.
#' @return A tibble: for threshold methods, `lag`, `statistic`,
#'   `studentized` and `kept`; otherwise the candidate column and `value`.
#' @export
tidy.tsorder_fit <- function(x, ...) {
  out <- x$path
  if (identical(x$family, "fh")) {
    out$kept <- abs(out$studentized) > x$tau
  }
  out
}

#' Glance at a tsorder fit
#'
#' @param x A [order_select()] result.
#' @param ... Unused.
#' @return A one-row tibble with `order`, `method`, `family`, `n` and
#'   (for threshold methods) `tau`.
#' @export
glance.tsorder_fit <- function(x, ...) {
  tibble::tibble(
    order = as.integer(x$order),
    method = x$method,
    family = x$family,
    n = as.integer(x$n),
    tau = if (is.null(x$tau)) NA_real_ else x$tau
  )
}

#' @importFrom tibble as_tibble
#' @export
tibble::as_tibble

#' Coerce a tsorder fit to a tibble
#'
#' The decision path as a tibble -- the same table as [tidy.tsorder_fit()],
#' provided so `as_tibble()` works directly on [order_select()] results.
#'
#' @param x A [order_select()] result.
#' @param ... Unused.
#' @return A tibble: for threshold methods, `lag`, `statistic`,
#'   `studentized` and `kept`; otherwise the candidate column and `value`.
#' @export
as_tibble.tsorder_fit <- function(x, ...) {
  tidy.tsorder_fit(x, ...)
}
