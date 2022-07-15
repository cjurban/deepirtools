#!/usr/bin/env Rscript

packages = c("mirt")
new_packages = packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages,
                   repos = "http://cran.us.r-project.org")
}

library(mirt)

args = commandArgs(trailingOnly = TRUE)
model_type = args[1]
sample_size = args[2]
expected_dir = args[3]
data_dir = args[4]

ldgs = read_csv(file.path(expected_dir, "ldgs.csv"), rownames_included = FALSE)
ints = read_csv(file.path(expected_dir, "ints.csv"), rownames_included = FALSE)
cov_mat = read_csv(file.path(expected_dir, "cov_mat.csv"), rownames_included = FALSE)

Y = simdata(a = ldgs,
            d = ints,
            N = sample_size,
            itemtype = if(model_type == "grm") {"graded"} else {model_type},
            sigma = cov_mat
           )
write_csv(Y, file.path(data_dir, "data.csv"), col_names = FALSE)