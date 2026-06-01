#'
#'  logilinquad.R
#'
#'  Quadrature schemes for logistic method for linear network

logi.dummy.linnet <- function(X, dummytype = "binomial", nd = NULL, mark.repeat = FALSE, ...) {
  library(spatstat.geom)
  library(spatstat.utils)
  
## Resolving nd inspired by default.n.tiling
  if(is.null(nd)){
    nd <- spatstat.options("ndummy.min")
    if(inherits(X, "lpp"))
      nd <- pmax(nd, 10 * ceiling(2 * sqrt(npoints(X))/10))
  }
  nd <- ensure2vector(nd)
  marx <- is.multitype(X)
  if(marx)
    lev <- levels(marks(X))
  if(marx && mark.repeat){
    N <- length(lev)
    Dlist <- inDlist <- vector("list", N)
  } else{
    N <- 1}
  
  type <- match.arg(dummytype, c("binomial", "poisson"))
  linnet <- as.linnet(X)
  B <- X$domain
  win <- as.psp(X)
  len <- lengths_psp(win)
  nseg <- length(len)
  ndumB <- nd[1L] * nd[2L]
  rho <- ndumB/sum(len)
  Dinfo <- list(nd=nd, rho=rho, how=type)

## Repeating dummy process for each mark type 1:N (only once if unmarked or mark.repeat = FALSE)
for(i in 1:N){
  switch(type,
         # stratrand={
         #   D <- as.lpp(stratrand(B, nd[1L], nd[2L]), W = B)
         #   inD <- which(inside.owin(D, w = W))
         #   D <- D[W]
         #   inD <- paste(i,inD,sep="_")
         # },
         binomial={
           D <- runiflpp(ndumB, linnet)
         },
         poisson={
           D <- runiflpp(rpois(1, ndumB), linnet)
         },
         stop("unknown dummy type"))
  if(marx && mark.repeat){
    marks(D) <- factor(lev[i], levels = lev)
    Dlist[[i]] <- D
  }
}

if(marx && mark.repeat){
  inD <- Reduce(append, inDlist)
  D <- Reduce(superimpose, Dlist)
}
# if(type %in% c("stratrand"))
#   Dinfo <- append(Dinfo, list(inD=inD))
if(marx && !mark.repeat){
  marks(D) <- sample(factor(lev, levels=lev), npoints(D), replace = TRUE)
  Dinfo$rho <- Dinfo$rho/length(lev)
}
attr(D, "dummy.parameters") <- Dinfo
return(D)
}

################################################################################
quadscheme.logi.linnet <- function(data, dummy, dummytype = "binomial", nd = NULL, mark.repeat = FALSE, ...){
  data <- as.lpp(data)
  ## If dummy is missing we generate dummy pattern with logi.dummy.
  if(missing(dummy))
    dummy <- logi.dummy.linnet(data, dummytype, nd, mark.repeat, ...)
  Dinfo <- attr(dummy, "dummy.parameters")
  D <- as.lpp(dummy)
  D_psp <- as.psp(D)
  len_D <- lengths_psp(D_psp)  
  if(is.null(Dinfo))
    Dinfo <- list(how="given", rho=npoints(D)/(sum(len_D)*markspace.integral(D)))

    ## Weights:
  n <- npoints(data)+npoints(D)
  w <- sum(lengths_psp(as.psp(window(data))))/n
  Q <- quad(data, D, rep(w,n), param=Dinfo)
  class(Q) <- c("logilinquad", class(Q))
  return(Q)
}

################################################################################
summary.logilinquad <- function(object, ..., checkdup=FALSE) {
  verifyclass(object, "logilinquad")
  s <- list(
    data  = summary.ppp(object$data, checkdup=checkdup),
    dummy = summary.ppp(object$dummy, checkdup=checkdup),
    param = object$param)
  class(s) <- "summary.logilinquad"
  return(s)
}

################################################################################
print.summary.logilinquad <- function(x, ..., dp=3) {
  cat("Quadrature scheme (logistic) = data + dummy\n")
  Dinfo <- x$param
  if(is.null(Dinfo))
    cat("created by an unknown function.\n")
  cat("Data pattern:\n")
  print(x$data, dp=dp)
  
  cat("\n\nDummy pattern:\n")
  # How they were computed
  switch(Dinfo$how,
         binomial={
           cat("(Binomial dummy points)\n")
         },
         poisson={
           cat("(Poisson dummy points)\n")
         },
         given=cat("(Dummy points given by user)\n")
  )
  # Description of them
  print(x$dummy, dp=dp)
  
  return(invisible(NULL))
}

