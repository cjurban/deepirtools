#!/usr/bin/env Rscript

packages <- c('argparse', "mirt")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages,
                   repos = "http://cran.us.r-project.org")
}