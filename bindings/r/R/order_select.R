#' Select the order of a time-series model
#'
#' The high-level interface of tsorder: one call that runs the chosen
#' selector and returns an object carrying the selected order and the whole
#' decision path (criterion values, or the studentised statistic and its
#' threshold), ready for printing and plotting.
#'
#' @param x A numeric vector (a stationary series; see
#'   [integration_order()] first if in doubt).
#' @param method What to run. Information criteria: `"bic"`, `"aic"`,
#'   `"aicc"`, `"hq"`, `"fpe"`, `"cat"`, `"mdl"`. Fine-Hwang threshold
#'   statistics: `"pacf"` (default), `"acf_bartlett"`, `"inverse_pacf"`,
#'   `"jirak"`, `"jirak_refit"` (Jirak's refitted q-hat(5)),
#'   `"pacf_robust"`, `"acf_robust"`, `"long_ar_ols"`. McMillan degree:
#'   `"svc"`, `"hankel_rank"`, `"hankel_svt"`.
#' @param ... Passed on to the underlying low-level function
#'   ([ic_order()], [fh_order()], [svc_order()], [hankel_rank_order()],
#'   [hankel_svt_order()]).
#' @return An object of class `"tsorder_fit"` with elements `order`,
#'   `method`, `family` (`"ic"`, `"fh"`, `"refit"` or `"mcmillan"`), `path` (a tibble
#'   of the decision path) and, for threshold methods, `tau`.
#' @examples
#' set.seed(1)
#' x <- arima.sim(list(ar = c(1.2, -0.6)), 400)
#' fit <- order_select(x)
#' fit
#' @export
order_select <- function(x, method = "pacf", ...) {
  ic <- c("bic", "aic", "aicc", "hq", "fpe", "cat", "mdl")
  fh <- c("pacf", "acf_bartlett", "inverse_pacf", "jirak", "pacf_robust",
          "acf_robust", "long_ar_ols", "long_ar_yw", "acf")
  mc <- c("svc", "hankel_rank", "hankel_svt")
  method <- rlang::arg_match(method, c(fh, ic, mc, "jirak_refit"))
  if (method == "jirak_refit") {
    r <- jirak_order(x, ...)
    path <- tibble::tibble(dimension = seq_along(r$per_dimension),
                           value = r$per_dimension)
    out <- list(order = r$order, method = method, family = "refit",
                path = path, tau = r$tau, p_n = r$p_n)
  } else if (method %in% ic) {
    r <- ic_order(x, criterion = method, ...)
    path <- tibble::tibble(order = seq_along(r$values) - 1L, value = r$values)
    out <- list(order = r$order, method = method, family = "ic",
                path = path, tau = NULL)
  } else if (method %in% fh) {
    stat <- if (method == "jirak") "jirak_band" else method
    args <- list(...)
    if (method == "jirak" && is.null(args$threshold)) args$threshold <- "gumbel"
    r <- do.call(fh_order, c(list(x = x, statistic = stat), args))
    path <- tibble::tibble(lag = seq_along(r$studentized),
                           statistic = r$values,
                           studentized = r$studentized)
    out <- list(order = r$order, method = method, family = "fh",
                path = path, tau = r$tau, p_n = r$p_n)
  } else {
    r <- switch(method,
      svc = svc_order(x, ...),
      hankel_rank = hankel_rank_order(x, ...),
      hankel_svt = hankel_svt_order(x, ...)
    )
    path <- switch(method,
      svc = tibble::tibble(order = seq_along(r$correlations),
                           value = r$correlations),
      hankel_rank = tibble::tibble(order = seq_along(r$pvalues) - 1L,
                                   value = r$pvalues),
      hankel_svt = tibble::tibble(order = seq_along(r$singular_values),
                                  value = r$singular_values)
    )
    out <- list(order = r$order, method = method, family = "mcmillan",
                path = path, tau = r$xi %||% NULL)
  }
  out$n <- length(x)
  out$call <- match.call()
  structure(out, class = "tsorder_fit")
}

#' @export
print.tsorder_fit <- function(x, ...) {
  what <- switch(x$family,
    ic = "order (argmin of the criterion)",
    fh = "order (last studentised statistic above the threshold)",
    refit = "order (max over the per-dimension refits)",
    mcmillan = "McMillan degree (state dimension)"
  )
  cli::cli_h1("tsorder fit - {x$method}")
  cli::cli_alert_success("Selected {what}: {.strong {x$order}}")
  if (!is.null(x$tau)) {
    cli::cli_alert_info("Threshold: {signif(x$tau, 4)}")
  }
  cli::cli_alert_info("n = {x$n}; path has {nrow(x$path)} entries - see {.code autoplot()} or {.code $path}")
  invisible(x)
}

#' Plot a tsorder fit
#'
#' For threshold methods, the studentised path with the threshold line and
#' the selected order highlighted; for criteria, the criterion values with
#' the argmin marked; for McMillan-degree methods, the decision path.
#'
#' @param object A [order_select()] result.
#' @param ... Unused.
#' @return A ggplot object.
#' @export
autoplot.tsorder_fit <- function(object, ...) {
  if (object$family == "fh") {
    df <- object$path
    df$kept <- abs(df$studentized) > object$tau
    ggplot2::ggplot(df, ggplot2::aes(.data$lag, abs(.data$studentized))) +
      ggplot2::geom_col(ggplot2::aes(fill = .data$kept), show.legend = FALSE) +
      ggplot2::geom_hline(yintercept = object$tau, linetype = 2) +
      ggplot2::scale_fill_manual(values = c(`TRUE` = "#2a78d6", `FALSE` = "#b8c4dc")) +
      ggplot2::labs(x = "lag", y = "|studentised statistic|",
                    title = sprintf("%s: order %d (threshold %.2f)",
                                    object$method, object$order, object$tau))
  } else {
    df <- object$path
    xvar <- names(df)[1]
    ggplot2::ggplot(df, ggplot2::aes(.data[[xvar]], .data$value)) +
      ggplot2::geom_line() +
      ggplot2::geom_point(data = df[df[[xvar]] == object$order, , drop = FALSE],
                          colour = "#2a78d6", size = 3) +
      ggplot2::labs(x = xvar, y = "value",
                    title = sprintf("%s: selected %d", object$method, object$order))
  }
}

#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' @importFrom rlang %||% .data
NULL
