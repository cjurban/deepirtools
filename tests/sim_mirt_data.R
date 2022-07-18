#!/usr/bin/env Rscript

rm(list = ls())

packages = c("mirt")
new_packages = packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages,
                   repos = "http://cran.us.r-project.org")
}

library(mirt)

args = commandArgs(trailingOnly = TRUE)
model_type = args[1]
sample_size = strtoi(args[2])
expected_dir = args[3]
data_dir = args[4]

ldgs = read.csv(file.path(expected_dir, "ldgs.csv"), header = FALSE)
ints = read.csv(file.path(expected_dir, "ints_R.csv"), header = FALSE)
cov_mat = read.csv(file.path(expected_dir, "cov_mat.csv"), header = FALSE)

if (model_type == "grm" & dim(ints)[2] > 1) {
  itemtype = ifelse(is.na(ints[, 2]), "2PL", "graded")
} else if (model_type == "grm" & dim(ints)[2] == 1) {
  itemtype = rep("2PL", dim(ints)[1])
} else if (model_type == "gpcm") {
  itemtype = rep("gpcm", dim(ints)[1])
  ints = cbind(rep(0, dim(ints)[1]), ints)
}

ldgs = matrix(as.vector(t(ldgs)), nrow = dim(ldgs)[1], byrow = TRUE)
ints = matrix(as.vector(t(ints)), nrow = dim(ints)[1], byrow = TRUE)
cov_mat = matrix(as.vector(t(cov_mat)), nrow = dim(cov_mat)[1], byrow = TRUE)

Y = simdata(a = ldgs,
            d = ints,
            N = sample_size,
            itemtype = itemtype,
            sigma = cov_mat
           )
write.table(Y, file = file.path(data_dir, "data.csv"), sep = ",", row.names = FALSE, col.names = FALSE)