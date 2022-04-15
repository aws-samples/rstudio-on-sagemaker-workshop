01\_Pipeline
================

# Use RStudio on SageMaker to create a SageMaker Pipeline and deploy to a serverless endpoint

This is an example of using RStudio on SageMaker showing how you can
create a SageMaker Pipeline with R as well as deploying your R model in
a serverless endpoint.

The following diagram shows the architecture used in this example.

![SMPipelinesRStudio](https://user-images.githubusercontent.com/18154355/163631036-88b6b936-b4f9-4228-815d-95611e65fdf0.png)

This file contains the logic to run the end to end process along with
comments for each of the steps.

## Folder/File Structure

-   `iam_policy.json` & `trust_relationship.json` contain the additional
    IAM policy and trust relationship that needs to be added to your
    assumed role and contain the permissions you will need to use
    CodeBuild to build the custom R containers
-   `docker/` contains the docker file definitions and helper files
    needed by the custom containers
-   `preprocessing`, `postprocessing` and `training_and_deploying`
    folders container the code for the respective steps
-   `pipeline.R` is the file containing the definition of the SageMaker
    Pipeline.

# Setup & preparation

To begin with, make sure you have all the necessary packages installed
and initialise some variables that we will need in the next cells.

``` r
suppressWarnings(library(dplyr))
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
suppressWarnings(library(reticulate))
suppressWarnings(library(readr))

sagemaker <- import('sagemaker')
boto3 <- import('boto3')

session <- sagemaker$Session()
bucket <- session$default_bucket()

role_arn <- sagemaker$get_execution_role()
account_id <- session$account_id()
region <- boto3$session$Session()$region_name

local_path <- dirname(rstudioapi::getSourceEditorContext()$path)
```

## Download data

For this example we will be using the famous abalone dataset as can be
found on the [UCI dataset
archive](https://archive.ics.uci.edu/ml/datasets/Abalone) where we will
create a model to predict the age of an abalone shell based on physical
measurements.

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
\[<http://archive.ics.uci.edu/ml>\]. Irvine, CA: University of
California, School of Information and Computer Science.

``` r
data_file <- 's3://sagemaker-sample-files/datasets/tabular/uci_abalone/abalone.csv'
data_string <- sagemaker$s3$S3Downloader$read_file(data_file)
abalone <- read_csv(file = data_string, col_names = FALSE)
```

    ## Rows: 4177 Columns: 9

    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (1): X1
    ## dbl (8): X2, X3, X4, X5, X6, X7, X8, X9

    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
names(abalone) <- c('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
head(abalone)
```

    ## # A tibble: 6 × 9
    ##   sex   length diameter height whole_weight shucked_weight viscera_weight
    ##   <chr>  <dbl>    <dbl>  <dbl>        <dbl>          <dbl>          <dbl>
    ## 1 M      0.455    0.365  0.095        0.514         0.224          0.101 
    ## 2 M      0.35     0.265  0.09         0.226         0.0995         0.0485
    ## 3 F      0.53     0.42   0.135        0.677         0.256          0.142 
    ## 4 M      0.44     0.365  0.125        0.516         0.216          0.114 
    ## 5 I      0.33     0.255  0.08         0.205         0.0895         0.0395
    ## 6 I      0.425    0.3    0.095        0.352         0.141          0.0775
    ## # … with 2 more variables: shell_weight <dbl>, rings <dbl>

``` r
dir.create(paste0(local_path,"/data"), showWarnings = FALSE)
write_csv(abalone, paste0(local_path,"/data/abalone_data.csv"))


s3_raw_data <- session$upload_data(path = paste0(local_path,"/data/abalone_data.csv"),
                                   bucket = bucket,
                                   key_prefix = 'pipeline-example/data')
```

We are also creating the variable `abalone_t`. This will be used for
testing the endpoint is available at a later stage.

``` r
abalone_t <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)
```

# Run pipeline

The pipelines is defined in the file `pipeline.R`. Head to that file to
dive deeper into how a SageMaker Pipeline is being defined. To run it,
we simply need to run the upsert method to create or update the pipeline
and then running the start method actually starts the execution of the
pipeline on SageMaker.

To view the pipeline as it is running, head to SageMaker Studio where a
custom UI will allow you to visualise the DAG of the execution of the
pipeline.

``` r
source(paste0(local_path, "/pipeline.R"))
my_pipeline <- get_pipeline(input_data_uri=s3_raw_data)

upserted <- my_pipeline$upsert(role_arn=role_arn)
execution <- my_pipeline$start()
```

``` r
execution$describe()
execution$wait()
```

# Deploy to serverless endpoint

Once the pipeline has finished running, a model will be registered to
the model registry and we will be able to deploy the model to an
endpoint. In this example we deploy on a serverless endpoint but you are
welcome to deploy with any of the supported deployment methods.

<br>

From all approved models in the model registry, we want to select the
one most recently created. We can simply query the model registry as
below to get the ARN for that model.

``` r
approved_models <- boto3$client("sagemaker")$list_model_packages(ModelApprovalStatus='Approved', 
                                                                 ModelPackageGroupName='AbaloneRModelPackageGroup',
                                                                 SortBy='CreationTime',
                                                                 SortOrder='Ascending')
model_package_arn <- approved_models[["ModelPackageSummaryList"]][[1]][["ModelPackageArn"]]
```

For the actual deployment, we need to create the SageMaker Model and
then we can use the SageMaker SDK to deploy to a serverless endpoint as
per below.

``` r
endpoint_name <- paste("serverless-r-abalone-endpoint", format(Sys.time(), '%Y%m%d-%H-%M-%S'), sep = '-')


model <- sagemaker$ModelPackage(role=role_arn, 
                                model_package_arn=model_package_arn, 
                                sagemaker_session=session)

serverless_config <- sagemaker$serverless$ServerlessInferenceConfig(memory_size_in_mb=1024L, max_concurrency=5L)
model$deploy(serverless_inference_config=serverless_config, endpoint_name=endpoint_name)
```

## Perform inference on test data

Using the data in variable `abalone_t` we will perform some sample
predictions using the newly deployed model to test that it is up and
running and capable of giving back predictions.

``` r
suppressWarnings(library(jsonlite))
x = list(features=format_csv(abalone_t[1:3,1:11]))
x = toJSON(x)

# test the endpoint
predictor <- sagemaker$predictor$Predictor(endpoint_name=endpoint_name, sagemaker_session=session)
predictor$predict(x)
```

    ## b'{"output":[9.312,7.8098,11.0181]}'

# Delete endpoint

Don’t forget to delete the running endpoint once you have finished
experimenting. In this case, if the endpoint remains unused, no costs
will be incurred, since it is a serverless endpoint, but nevertheless it
is a good practice to always shut down unused resources/endpoints at the
end of experimentation.

``` r
predictor$delete_endpoint(delete_endpoint_config=TRUE)
```
