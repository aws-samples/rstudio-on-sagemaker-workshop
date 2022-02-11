# Bring in library for prediction server
library(plumber)
library(mda)

# Setup scoring function
serve <- function() {
  app <- plumb('/opt/ml/plumber.R')
  app$run(host='0.0.0.0', port=8080)}

serve()
