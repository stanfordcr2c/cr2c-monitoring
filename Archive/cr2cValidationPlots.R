# Show all error messages
options(show.error.messages = TRUE)
# Supress warnings to not freak people out!
options(warn = -1)

# Helper function to install and load package if not on machine this is running on
packages=function(pckg_nm){
  x=as.character(match.call()[[2]])
  if (!require(pckg_nm,character.only=TRUE)){
    install.packages(pkgs=pckg_nm,repos="http://cran.r-project.org")
    require(pckg_nm,character.only=TRUE)
  }
}
packages(stepR)
library(stepR)
packages(magrittr)
library(magrittr)

options(echo=TRUE)
args = commandArgs(trailingOnly = TRUE)

print(args)

