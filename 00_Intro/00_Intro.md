00\_Intro
================

## RStudio on SageMaker Introduction

In collaboration with RStudio PBC, we are excited to announce the
general availability of RStudio on Amazon SageMaker, the industry’s
first fully managed RStudio Workbench IDE in the cloud. RStudio on
SageMaker provides the familiar IDE that is known and loved throughout
the R community.

### Benefits on SageMaker

RStudio on SageMaker establishes user authentication through IAM or SSO.
Once authenticated the user assumes their SageMaker execution role which
has granular permissions for all AWS functionality.

This means that once authenticated, you can access S3 datasets, train
and host models using SageMaker, launch AWS Glue jobs, etc without the
need to re-authenticate yourself within the IDE.

Additionally, among many of other benefits, you can right size the
instance backing your RStudio session, and use the full flexibility of
the cloud.

### User EFS Mount

When on-boarding a UserProfile to a SageMaker domain, a home directory
is added to the Domains EFS (Network) storage. This is your personal
storage location where can put code repositories, datasets, and other
file objects. You can see this EFS mount as your Home directory within
the RStudio IDE panel.

### Right IDE at the right time

This EFS home is shared across the Studio IDE you choose. In other
words, you can utilize Studio’s Jupyter or RStudio IDE with access to
the same datasets and code repositories.

### Terminal

Within your RStudio Session, you have access to the terminal within your
container and can make OS level installs / utilize command line programs
like `git`.

## Data Access

There are several methods to access data within the RStudio on SageMaker
IDE.

### Download to EFS

Using OS tooling

``` bash
mkdir -p ./dataset/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -O ./dataset/abalone.csv
```

Using aws cli

``` bash
aws s3 cp s3://sagemaker-sample-files/datasets/tabular/uci_abalone/abalone.csv ./dataset/
```

    ## Completed 187.4 KiB/187.4 KiB (320.0 KiB/s) with 1 file(s) remainingdownload: s3://sagemaker-sample-files/datasets/tabular/uci_abalone/abalone.csv to dataset/abalone.csv

### Utilize Native R Packages to read from Disk or HTTP

``` r
if (!'tidyverse' %in% installed.packages()) {install.packages('tidyverse')}
suppressWarnings(library(tidyverse))
```

``` r
df_http <- read_csv(file = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', show_col_types = FALSE)
df_disk <- read_csv(file = 'dataset/abalone.csv', show_col_types = FALSE)
head(df_http)
```

    ## # A tibble: 6 × 9
    ##   M     `0.455` `0.365` `0.095` `0.514` `0.2245` `0.101` `0.15`  `15`
    ##   <chr>   <dbl>   <dbl>   <dbl>   <dbl>    <dbl>   <dbl>  <dbl> <dbl>
    ## 1 M       0.35    0.265   0.09    0.226   0.0995  0.0485  0.07      7
    ## 2 F       0.53    0.42    0.135   0.677   0.256   0.142   0.21      9
    ## 3 M       0.44    0.365   0.125   0.516   0.216   0.114   0.155    10
    ## 4 I       0.33    0.255   0.08    0.205   0.0895  0.0395  0.055     7
    ## 5 I       0.425   0.3     0.095   0.352   0.141   0.0775  0.12      8
    ## 6 F       0.53    0.415   0.15    0.778   0.237   0.142   0.33     20

### Utilize Python Boto3 or SageMaker SDK with Reticulate

First, load the `reticulate` library and import the `sagemaker` Python
module. Once the module is loaded, use the `$` notation in R instead of
the `.` notation in Python to use available classes.

The reticulate and python SDKs come pre-installed in the RStudio on
SageMaker containers.

``` r
# Packages ----
suppressWarnings(library(reticulate))

# Python packages ----
sagemaker <- import("sagemaker")
class(sagemaker)
```

    ## [1] "python.builtin.module" "python.builtin.object"

Let’s create an Amazon Simple Storage Service (S3) bucket for your data.

``` r
session <- sagemaker$Session()
bucket <- session$default_bucket()
print(bucket)
```

Upload data to personal S3 bucket

``` r
abalone_on_s3_uri <- session$upload_data(path = 'dataset/abalone.csv', bucket = bucket, key_prefix = 'data')
print(abalone_on_s3_uri)
```

### Utilize Native R Packages to read from S3

The `aws.s3` library provides a `s3read_using` function to load data
directly into memory. Using the additional `aws.ec2metadata` library, we
are able to utilize your SageMaker execution role’s credentials.

``` r
if (!'aws.s3' %in% installed.packages()) {install.packages('aws.s3')}
if (!'aws.ec2metadata' %in% installed.packages()) {install.packages('aws.ec2metadata')}

suppressWarnings(library(aws.s3))
df_s3 <- s3read_using(FUN = read.csv, object = "data/abalone.data", bucket = bucket)
head(df_s3)
```

    ##   M X0.455 X0.365 X0.095 X0.514 X0.2245 X0.101 X0.15 X15
    ## 1 M  0.350  0.265  0.090 0.2255  0.0995 0.0485 0.070   7
    ## 2 F  0.530  0.420  0.135 0.6770  0.2565 0.1415 0.210   9
    ## 3 M  0.440  0.365  0.125 0.5160  0.2155 0.1140 0.155  10
    ## 4 I  0.330  0.255  0.080 0.2050  0.0895 0.0395 0.055   7
    ## 5 I  0.425  0.300  0.095 0.3515  0.1410 0.0775 0.120   8
    ## 6 F  0.530  0.415  0.150 0.7775  0.2370 0.1415 0.330  20

## Package Management

Users are able to install packages using the native R `install` command
as well as through the graphical interface in Rstudio. When creating
your domain there is an optional parameter to set a RStudio Package
Manager URL so your team can utilize internal repositories as well.

## Publishing to RStudio Connect

Functionality to publishing to RStudio Connect works as expected and
depending on your networking configuration, your domain can utilize
RStudio connect servers in a private subnet.
