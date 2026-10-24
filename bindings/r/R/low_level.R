#' Information-criterion order selection
#'
#' Selects an AR order by minimising a penalised-likelihood criterion over
#' nested fits, with the innovation variances estimated by Yule-Walker
#' (one Levinson-Durbin pass, the convention of [stats::ar.yw()]), Burg, or
#' conditional least squares on a common sample.
#'
#' @param x A numeric vector (a stationary series).
#' @param criterion One of `"aic"`, `"aicc"`, `"bic"`, `"hq"`, `"fpe"`,
#'   `"cat"`, `"mdl"`.
#' @param method One of `"yw"` (default), `"burg"`, `"ols"`.
#' @param max_order Maximum candidate order; `NULL` uses the [stats::ar()]
#'   default `min(n - 1, 10 log10 n)`.
#' @return A list with `order`, the criterion `values` for orders
#'   `0:max_order`, and the innovation variances `sigma2`.
#' @examples
#' set.seed(1)
#' x <- arima.sim(list(ar = c(1.2, -0.6)), 400)
#' ic_order(x, "bic")$order
#' @export
ic_order <- function(x, criterion = "bic", method = "yw", max_order = NULL) {
  rs_ic_order(as.double(x), criterion, method,
              if (is.null(max_order)) NULL else as.integer(max_order))
}

#' Fine-Hwang threshold order selection
#'
#' Fits a model whose memory `p_n` grows with the sample size, studentises
#' the chosen statistic, clips everything below the threshold `tau_n`, and
#' reports the last surviving lag (Fine & Hwang 1979).
#'
#' @inheritParams ic_order
#' @param statistic One of `"pacf"` (AR, default), `"acf_bartlett"` (MA),
#'   `"inverse_pacf"` (MA), `"jirak_band"` (AR, exact Jirak-2012
#'   studentisation), `"pacf_robust"` and `"acf_robust"` (experimental,
#'   25% breakdown against additive outliers), `"long_ar_yw"`,
#'   `"long_ar_ols"`, `"acf"`.
#' @param threshold One of `"bic"` (default, `sqrt(c log n)`), `"lil"`,
#'   `"maxk"`, `"gumbel"` (level `alpha`), `"fixed"` (value `c`).
#' @param c Threshold constant (default 1.5).
#' @param alpha Level for the `"gumbel"` threshold (default 0.05).
#' @param memory Growing memory `p_n`; `NULL` uses the statistic's default
#'   (`ceiling(2 log n)`).
#' @return A list with `order`, `p_n`, `tau`, the raw `values` and the
#'   `studentized` path.
#' @examples
#' set.seed(1)
#' x <- arima.sim(list(ar = c(1.2, -0.6)), 400)
#' fh_order(x)$order
#' @export
fh_order <- function(x, statistic = "pacf", threshold = "bic", c = 1.5,
                     alpha = 0.05, memory = NULL) {
  rs_fh_order(as.double(x), statistic, threshold, c, alpha,
              if (is.null(memory)) NULL else as.integer(memory))
}

#' Jirak's refitted AR order estimator
#'
#' Jirak's (2012) q-hat(5): fits Yule-Walker AR(k) for every dimension
#' `k <= p_n`, studentises each coefficient vector with its exact null
#' standard error, applies a common Gumbel threshold calibrated on the
#' maximal dimension, and returns the largest per-dimension last-crossing.
#' The refits recover power that the single full-dimension band
#' (`fh_order(statistic = "jirak_band", threshold = "gumbel")`) can lose
#' when a coefficient is small in the long-AR representation.
#'
#' @inheritParams ic_order
#' @param alpha Level of the Gumbel band (default 0.05); a slowly vanishing
#'   level gives the consistent variant.
#' @param memory Maximal fitting dimension `p_n`; `NULL` uses the default
#'   `ceiling(2 log n)`.
#' @return A list with `order` (the maximum), `p_n`, `tau` and the
#'   `per_dimension` estimates for `k = 1:p_n`.
#' @examples
#' set.seed(1)
#' x <- arima.sim(list(ar = c(1.2, -0.6)), 400)
#' jirak_order(x)$order
#' @export
jirak_order <- function(x, alpha = 0.05, memory = NULL) {
  rs_jirak_order(as.double(x), alpha,
                 if (is.null(memory)) NULL else as.integer(memory))
}

#' McMillan degree by Bauer's SVC criterion
#'
#' Estimates the state dimension from the past-future sample canonical
#' correlations (CCA weighting), minimising
#' `SVC(n) = sigma_{n+1}^2 + C(T) 2n / T` (Bauer 2001). Structurally immune
#' to observation noise, which changes the ARMA orders but not the McMillan
#' degree.
#'
#' @inheritParams ic_order
#' @param penalty `"logt"` (default, Bauer's practical choice) or `"fplogt"`
#'   (his consistent penalty; strongly underestimates in finite samples).
#' @param window_factor The window is `f = p = floor(window_factor *
#'   p_AIC)`; Bauer recommends values in `[1, 2]`.
#' @param window Explicit window overriding the rule.
#' @return A list with `order`, the canonical `correlations`, the criterion
#'   `values` and the window `f`, `p`.
#' @export
svc_order <- function(x, penalty = "logt", window_factor = 1, window = NULL) {
  rs_svc_order(as.double(x), penalty, window_factor,
               if (is.null(window)) NULL else as.integer(window))
}

#' McMillan degree by a sequential rank test on the autocovariance Hankel
#'
#' Robin & Smith's (2000) characteristic-root statistic with the weighted
#' chi-squared null evaluated by Imhof's exact inversion; tests ranks
#' r = 0, 1, ... and stops at the first non-rejection, at a level that
#' vanishes with the sample size (consistent by Hosoya's condition).
#'
#' @inheritParams ic_order
#' @param alpha Fixed test level, or `NULL` (default) for the vanishing
#'   level `0.05 log(50) / log(n)`.
#' @param k Hankel dimension; `NULL` uses `max(3, ceiling(log n))`.
#' @return A list with `order`, the `pvalues` of the stages run, `alpha`
#'   and `k`.
#' @export
hankel_rank_order <- function(x, alpha = NULL, k = NULL) {
  rs_hankel_rank_order(as.double(x), alpha,
                       if (is.null(k)) NULL else as.integer(k))
}

#' McMillan degree by Hankel singular-value thresholding
#'
#' Cuts the singular values of the autocovariance Hankel at a calibrated
#' threshold carrying the scale of the estimation error (after Zheng, Jedra
#' & Proutiere 2025). Recovery requires separation: the smallest non-zero
#' singular value of the true Hankel must exceed the cut.
#'
#' @inheritParams hankel_rank_order
#' @param c Threshold constant (default 1).
#' @return A list with `order`, the `singular_values`, the threshold `xi`
#'   and `k`.
#' @export
hankel_svt_order <- function(x, c = 1, k = NULL) {
  rs_hankel_svt_order(as.double(x), c, if (is.null(k)) NULL else as.integer(k))
}

#' Extended sample autocorrelation function
#'
#' Tsay & Tiao's (1984) EACF table (entries match `TSA::eacf` digit for
#' digit) with an automated reading of the zero-triangle vertex, which
#' names the ARMA order (p, q).
#'
#' @inheritParams ic_order
#' @param max_ar,max_ma Table dimensions (defaults 7 and 13, as in
#'   `TSA::eacf`).
#' @param band Band multiplier for the X/O coding (default 2).
#' @return A list with `p`, `q`, `found`, the `table` and the 0/1
#'   `significant` matrix.
#' @export
eacf <- function(x, max_ar = 7, max_ma = 13, band = 2) {
  rs_eacf(as.double(x), as.integer(max_ar), as.integer(max_ma), band)
}

#' Hannan-Rissanen ARMA order selection
#'
#' Stages 1-2 of Hannan & Rissanen (1982): innovations from a long AIC
#' autoregression, then BIC over regressions on lagged values and lagged
#' innovations. By default stage 2 is repeated on the recursive residuals
#' of the first-pass model - the consistency fix of their 1983 correction,
#' without which the selection does not converge when the MA polynomial has
#' zeros near the unit circle.
#'
#' @inheritParams ic_order
#' @param max_p,max_q Candidate grid (defaults 5 and 5).
#' @param erratum Apply the 1983 correction pass (default `TRUE`).
#' @return A list with `p`, `q`, the stage-2 coefficient estimates `phi`
#'   and `theta`, `sigma2` and the long-AR order used.
#' @export
hannan_rissanen <- function(x, max_p = 5, max_q = 5, erratum = TRUE) {
  rs_hannan_rissanen(as.double(x), as.integer(max_p), as.integer(max_q), erratum)
}

#' Order of integration by sequential unit-root testing
#'
#' Differences the series until the unit-root indication disappears, at a
#' level that vanishes with the sample size so the number of differences is
#' estimated consistently. KPSS (null = stationarity) by default; ADF with
#' BIC-chosen augmentation as an alternative. Critical values come from
#' embedded asymptotic quantile tables validated against the published
#' points.
#'
#' @inheritParams hankel_rank_order
#' @param max_d Largest d considered (default 2).
#' @param test `"kpss"` (default) or `"adf"`.
#' @return A list with `d`, the stage `stats` and `crits`, `alpha`, `test`.
#' @export
integration_order <- function(x, max_d = 2, alpha = NULL, test = "kpss") {
  rs_integration_order(as.double(x), as.integer(max_d), alpha, test)
}

#' Q_n robust scale estimator
#'
#' Rousseeuw & Croux's Q_n by the O(n log n) algorithm, with the constant
#' and finite-sample corrections of `robustbase::Qn` (the two agree to
#' near machine precision).
#'
#' @inheritParams ic_order
#' @return The scale estimate (a length-one numeric).
#' @export
qn <- function(x) rs_qn(as.double(x))

#' Robust autocorrelations
#'
#' Ma & Genton's (2000) Q_n-based autocorrelations in the bounded ratio
#' form, whose shared denominator cancels the additive-outlier bias.
#'
#' @inheritParams ic_order
#' @param max_lag Largest lag.
#' @return Autocorrelations for lags `0:max_lag`.
#' @export
robust_acf <- function(x, max_lag) rs_robust_acf(as.double(x), as.integer(max_lag))
