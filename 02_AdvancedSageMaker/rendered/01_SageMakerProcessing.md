01\_SageMakerProcessing
================

Clear the workspace

``` r
rm(list = ls())
```

## SageMaker Processing

## Imports

``` r
suppressWarnings(library(reticulate))
sagemaker <- import('sagemaker')
```

``` r
role = sagemaker$get_execution_role()
session = sagemaker$Session()
s3_output = session$default_bucket()
s3_prefix = "R-in-Processing"

account_id <- session$account_id()
region <- session$boto_region_name
```

``` r
container_uri <- paste(account_id, "dkr.ecr", region, "amazonaws.com/sagemaker-r-processing:1.0", sep=".")
print(container_uri)
```

``` r
processor <- sagemaker$processing$ScriptProcessor(image_uri = container_uri,
                                                   command=list("Rscript"),
                                                   role = role,
                                                   instance_count=1L,
                                                   instance_type="ml.c5.xlarge")
processor$run(
    code="/home/sagemaker-user/rstudio-on-sagemaker-workshop/02_AdvancedSageMaker/preprocessing.R",
    job_name=paste("r-processing", as.integer(as.numeric(Sys.time())), sep="-"))
```
