01\_SimpleUseCase
================

# Simple RStudio Use Case

Clear the workspace

``` r
rm(list = ls())
```

``` r
if (!'tidyverse' %in% installed.packages()) {install.packages('tidyverse')}
if (!'tidymodels' %in% installed.packages()) {install.packages('tidymodels')}
suppressWarnings(library(tidyverse))
suppressWarnings(library(tidymodels))
```

## Downloading and Processing the Dataset

The model uses the [abalone
dataset](https://archive.ics.uci.edu/ml/datasets/abalone) from the UCI
Machine Learning Repository.

``` r
abalone <- read_csv(file = 'dataset/abalone.csv', col_names = FALSE)
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

The output above shows that `sex` is a factor data type but is currently
a character data type (F is Female, M is male, and I is infant). Change
`sex` to a factor and view the statistical summary of the dataset:

``` r
abalone <- abalone %>%
  mutate(sex = as_factor(sex))
```

``` r
summary(abalone)
```

    ##  sex          length         diameter          height        whole_weight   
    ##  M:1528   Min.   :0.075   Min.   :0.0550   Min.   :0.0000   Min.   :0.0020  
    ##  F:1307   1st Qu.:0.450   1st Qu.:0.3500   1st Qu.:0.1150   1st Qu.:0.4415  
    ##  I:1342   Median :0.545   Median :0.4250   Median :0.1400   Median :0.7995  
    ##           Mean   :0.524   Mean   :0.4079   Mean   :0.1395   Mean   :0.8287  
    ##           3rd Qu.:0.615   3rd Qu.:0.4800   3rd Qu.:0.1650   3rd Qu.:1.1530  
    ##           Max.   :0.815   Max.   :0.6500   Max.   :1.1300   Max.   :2.8255  
    ##  shucked_weight   viscera_weight    shell_weight        rings       
    ##  Min.   :0.0010   Min.   :0.0005   Min.   :0.0015   Min.   : 1.000  
    ##  1st Qu.:0.1860   1st Qu.:0.0935   1st Qu.:0.1300   1st Qu.: 8.000  
    ##  Median :0.3360   Median :0.1710   Median :0.2340   Median : 9.000  
    ##  Mean   :0.3594   Mean   :0.1806   Mean   :0.2388   Mean   : 9.934  
    ##  3rd Qu.:0.5020   3rd Qu.:0.2530   3rd Qu.:0.3290   3rd Qu.:11.000  
    ##  Max.   :1.4880   Max.   :0.7600   Max.   :1.0050   Max.   :29.000

The summary above shows that the minimum value for `height` is 0.

Visually explore which abalones have `height` equal to 0 by plotting the
relationship between `rings` and `height` for each value of `sex`:

``` r
abalone %>%
  ggplot(aes(x = height, y = rings, color = sex)) +
  geom_point() +
  geom_jitter() +
  theme_minimal()
```

![](/home/sagemaker-user/rstudio-on-sagemaker-workshop/00_Intro/rendered/01_SimpleUseCase_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

The plot shows multiple outliers: two infant abalones with a height of 0
and a few female and male abalones with greater heights than the rest.
Let’s filter out the two infant abalones with a height of 0.

``` r
abalone <- abalone %>%
  filter(height != 0)

head(abalone)
```

    ## # A tibble: 6 × 9
    ##   sex   length diameter height whole_weight shucked_weight viscera_weight
    ##   <fct>  <dbl>    <dbl>  <dbl>        <dbl>          <dbl>          <dbl>
    ## 1 M      0.455    0.365  0.095        0.514         0.224          0.101 
    ## 2 M      0.35     0.265  0.09         0.226         0.0995         0.0485
    ## 3 F      0.53     0.42   0.135        0.677         0.256          0.142 
    ## 4 M      0.44     0.365  0.125        0.516         0.216          0.114 
    ## 5 I      0.33     0.255  0.08         0.205         0.0895         0.0395
    ## 6 I      0.425    0.3    0.095        0.352         0.141          0.0775
    ## # … with 2 more variables: shell_weight <dbl>, rings <dbl>

## Preparing the Dataset for Model Training

The model needs three datasets: one for training, testing, and
validation. First, convert `sex` into a dummy variable and move the
target, `rings`, to the first column. Amazon SageMaker algorithm require
the target to be in the first column of the dataset.

``` r
abalone <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)
abalone <- abalone %>%
  select(rings:infant, length:shell_weight)
head(abalone)
```

    ## # A tibble: 6 × 11
    ##   rings female  male infant length diameter height whole_weight shucked_weight
    ##   <dbl>  <int> <int>  <int>  <dbl>    <dbl>  <dbl>        <dbl>          <dbl>
    ## 1    15      0     1      0  0.455    0.365  0.095        0.514         0.224 
    ## 2     7      0     1      0  0.35     0.265  0.09         0.226         0.0995
    ## 3     9      1     0      0  0.53     0.42   0.135        0.677         0.256 
    ## 4    10      0     1      0  0.44     0.365  0.125        0.516         0.216 
    ## 5     7      0     0      1  0.33     0.255  0.08         0.205         0.0895
    ## 6     8      0     0      1  0.425    0.3    0.095        0.352         0.141 
    ## # … with 2 more variables: viscera_weight <dbl>, shell_weight <dbl>

Sample data for model training

``` r
set.seed(42)
abalone_train <- abalone %>%
  sample_frac(size = 0.7)
abalone <- anti_join(abalone, abalone_train)
```

    ## Joining, by = c("rings", "female", "male", "infant", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight")

``` r
abalone_test <- abalone %>%
  sample_frac(size = 0.5)
abalone_valid <- anti_join(abalone, abalone_test)
```

    ## Joining, by = c("rings", "female", "male", "infant", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight")

``` r
lm_model <- linear_reg() %>%
            set_engine('lm') %>% 
            set_mode('regression')
```

``` r
lm_fit <- lm_model %>%
          fit(rings ~ ., data = abalone_train)
```

``` r
summary(lm_fit$fit)
```

    ## 
    ## Call:
    ## stats::lm(formula = rings ~ ., data = data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8.2769 -1.2959 -0.3220  0.8369 13.9961 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)      3.1859     0.3203   9.947  < 2e-16 ***
    ## female           0.9442     0.1218   7.752 1.24e-14 ***
    ## male             0.9588     0.1132   8.467  < 2e-16 ***
    ## infant               NA         NA      NA       NA    
    ## length          -0.6564     2.1514  -0.305     0.76    
    ## diameter        11.6201     2.6494   4.386 1.20e-05 ***
    ## height           7.9790     1.6630   4.798 1.68e-06 ***
    ## whole_weight     8.2711     0.8607   9.610  < 2e-16 ***
    ## shucked_weight -19.1677     0.9764 -19.632  < 2e-16 ***
    ## viscera_weight  -9.6986     1.5505  -6.255 4.56e-10 ***
    ## shell_weight    10.0007     1.3180   7.588 4.36e-14 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2.183 on 2912 degrees of freedom
    ## Multiple R-squared:  0.5481, Adjusted R-squared:  0.5467 
    ## F-statistic: 392.5 on 9 and 2912 DF,  p-value: < 2.2e-16

``` r
test_results <- predict(lm_fit, new_data = abalone_test) %>%
                            bind_cols(abalone_test)
```

    ## Warning in predict.lm(object = object$fit, newdata = new_data, type =
    ## "response"): prediction from a rank-deficient fit may be misleading

``` r
head(test_results)
```

    ## # A tibble: 6 × 12
    ##   .pred rings female  male infant length diameter height whole_weight
    ##   <dbl> <dbl>  <int> <int>  <int>  <dbl>    <dbl>  <dbl>        <dbl>
    ## 1 10.7     10      0     1      0  0.595    0.48   0.185        1.18 
    ## 2  8.10     7      0     0      1  0.44     0.33   0.11         0.370
    ## 3 12.0     13      0     1      0  0.65     0.515  0.18         1.33 
    ## 4 10.1     11      0     0      1  0.52     0.38   0.14         0.525
    ## 5 13.3     12      1     0      0  0.61     0.495  0.185        1.11 
    ## 6 10.6     10      1     0      0  0.485    0.375  0.135        0.556
    ## # … with 3 more variables: shucked_weight <dbl>, viscera_weight <dbl>,
    ## #   shell_weight <dbl>

``` r
ggplot(data = test_results,
       mapping = aes(x = .pred, y = rings)) +
  geom_point(color = '#006EA1') +
  geom_abline(intercept = 0, slope = 1, color = 'orange') +
  labs(title = 'Linear Regression Results',
       x = 'Predicted Rings',
       y = 'Actual Rings')
```

![](/home/sagemaker-user/rstudio-on-sagemaker-workshop/00_Intro/rendered/01_SimpleUseCase_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->
