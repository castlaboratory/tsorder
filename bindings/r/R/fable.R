#' tsorder model for the fable/tidyverts framework
#'
#' A `fabletools` model definition: selects the ARIMA orders with a tsorder
#' selector (via [fit_arima()]) and fits [stats::arima()], so tsorder plugs
#' directly into `fabletools::model()`, `forecast()`, `accuracy()` and
#' friends alongside `fable::ARIMA()`.
#'
#' Requires the suggested packages `fabletools`, `tsibble` and
#' `distributional`.
#'
#' @param formula Model specification (the response; exogenous regressors
#'   are not supported).
#' @param method As in [fit_arima()]: an [order_select()] method or `"hr"`
#'   (default) for Hannan-Rissanen, which selects `p` and `q` jointly.
#' @param ... Passed on to [fit_arima()] (e.g. `d`, selector options).
#' @return A model definition usable inside `fabletools::model()`.
#' @examples
#' \dontrun{
#' library(fabletools)
#' as_tsibble(USAccDeaths) |>
#'   model(tsorder = TSORDER(value)) |>
#'   forecast(h = 12)
#' }
#' @export
TSORDER <- function(formula, method = "hr", ...) {
  if (!requireNamespace("fabletools", quietly = TRUE)) {
    cli::cli_abort("{.fn TSORDER} requires the {.pkg fabletools} package.")
  }
  model <- fabletools::new_model_class(
    "TSORDER",
    train = train_tsorder,
    specials = make_specials_tsorder()
  )
  fabletools::new_model_definition(model, !!rlang::enquo(formula),
                                   method = method, ...)
}

make_specials_tsorder <- function() {
  fabletools::new_specials(
    xreg = function(...) {
      cli::cli_abort("Exogenous regressors are not supported by {.fn TSORDER}.")
    }
  )
}

train_tsorder <- function(.data, specials, method = "hr", ...) {
  y <- as.numeric(.data[[tsibble::measured_vars(.data)[1]]])
  if (anyNA(y)) {
    cli::cli_abort("{.fn TSORDER} does not support missing values.")
  }
  fit <- fit_arima(y, method = method, ...)
  structure(
    list(fit = fit, y = y, method = method),
    class = "fbl_tsorder"
  )
}

#' @exportS3Method generics::forecast
forecast.fbl_tsorder <- function(object, new_data, ...) {
  h <- nrow(new_data)
  p <- stats::predict(object$fit, n.ahead = h)
  distributional::dist_normal(as.numeric(p$pred), as.numeric(p$se))
}

#' @exportS3Method stats::fitted
fitted.fbl_tsorder <- function(object, ...) {
  object$y - as.numeric(stats::residuals(object$fit))
}

#' @exportS3Method stats::residuals
residuals.fbl_tsorder <- function(object, ...) {
  as.numeric(stats::residuals(object$fit))
}

#' @exportS3Method fabletools::model_sum
model_sum.fbl_tsorder <- function(x) {
  ord <- x$fit$arma[c(1, 6, 2)]
  sprintf("TSORDER(%d,%d,%d)", ord[1], ord[2], ord[3])
}

#' @exportS3Method generics::glance
glance.fbl_tsorder <- function(x, ...) {
  tibble::tibble(
    sigma2 = x$fit$sigma2,
    log_lik = as.numeric(stats::logLik(x$fit)),
    AIC = x$fit$aic,
    method = x$method
  )
}

#' @exportS3Method generics::tidy
tidy.fbl_tsorder <- function(x, ...) {
  co <- stats::coef(x$fit)
  se <- if (length(co)) sqrt(pmax(diag(x$fit$var.coef), 0)) else numeric()
  tibble::tibble(term = names(co), estimate = as.numeric(co), std.error = se)
}
