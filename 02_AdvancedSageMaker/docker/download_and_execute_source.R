# TEST R Download

# Bring in library that allows parsing of JSON training parameters
library(jsonlite)
library(reticulate)
library(stringr)

boto3 <- import('boto3')
s3 <- boto3$client('s3')

# Setup parameters
# Container directories
prefix <- '/opt/ml'
input_path <- paste(prefix, 'input/data', sep='/')
output_path <- paste(prefix, 'output', sep='/')
model_path <- paste(prefix, 'model', sep='/')
code_dir <- paste(prefix, 'code', sep='/')


# This is where the hyperparamters are saved by the estimator on the container instance
param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')
params <- read_json(param_path)

s3_source_code_tar <- gsub('"', '', params$sagemaker_submit_directory)
script <- gsub('"', '', params$sagemaker_program)

bucketkey <- str_replace(s3_source_code_tar, "s3://", "")
bucket <- str_remove(bucketkey, "/.*")
key <- str_remove(bucketkey, ".*?/")

s3$download_file(bucket, key, "sourcedir-test.tar.gz")
untar("sourcedir-test.tar.gz", exdir=code_dir)

source(file.path(code_dir, script))
train()

