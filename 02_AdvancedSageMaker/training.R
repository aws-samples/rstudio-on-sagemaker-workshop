# Bring in library that allows parsing of JSON training parameters
library(jsonlite)

# Bring in library that contains multivariate adaptive regression splines (MARS)
library(mda)


# Setup parameters
# Container directories
prefix <- '/opt/ml'
input_path <- paste(prefix, 'input/data', sep='/')
output_path <- paste(prefix, 'output', sep='/')
model_path <- paste(prefix, 'model', sep='/')

# This is where the hyperparamters are saved by the estimator on the container instance
param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')

# Setup training function
train <- function() {
  
  training_path <- paste(input_path, "train", sep='/')
  
  # Read in hyperparameters
  training_params <- read_json(param_path)
  
  # Setting the target
  target <- 'Sepal.Length'
  
  if (!is.null(training_params$degree)) {
    degree <- as.numeric(training_params$degree)}
  else {
    degree <- 2}
  
  if (!is.null(training_params$thresh)) {
    thresh <- as.numeric(training_params$thresh)}
  else {
    thresh <- 0.001}
  
  if (!is.null(training_params$prune)) {
    prune <- as.logical(training_params$prune)}
  else {
    prune <- TRUE}
  
  # Bring in data
  training_files = list.files(path=training_path, full.names=TRUE)
  training_data = do.call(rbind, lapply(training_files, read.csv))
  
  # Convert to model matrix
  training_X <- model.matrix(~., training_data[, colnames(training_data) != target])
  
  # Save factor levels for scoring
  factor_levels <- lapply(training_data[, sapply(training_data, is.factor), drop=FALSE],
                          function(x) {levels(x)})
  
  # Run multivariate adaptive regression splines algorithm
  model <- mars(x=training_X, y=training_data[, target], degree=degree, thresh=thresh, prune=prune)
  
  # Generate outputs
  mars_model <- model[!(names(model) %in% c('x', 'residuals', 'fitted.values'))]
  attributes(mars_model)$class <- 'mars'
  save(mars_model, factor_levels, file=paste(model_path, 'mars_model.RData', sep='/'))
  print(summary(mars_model))
  print(paste('gcv:', mars_model$gcv))
  print(paste('mse:', sum((model$fitted.values - training_data[, target]) ** 2)))
  
  write('success', file=paste(output_path, 'success', sep='/'))
}
