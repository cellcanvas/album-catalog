# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2024-08-11
Remove defaults

## [0.1.3] - 2024-05-01
Fix embedding aggregation

## [0.1.1] - 2024-05-01
Update for aggregating multiple embeddings

## [0.0.9] - 2024-04-30
Use bounding boxes to skip regions

## [0.0.8] - 2024-04-30
Fix duplicate run functions

## [0.0.7] - 2024-04-30
Back to processing per location

## [0.0.6] - 2024-04-30
Manually serialize with dill

## [0.0.5] - 2024-04-30
Use dill for parallelization

## [0.0.4] - 2024-04-30
Add support for batches of locatoins

## [0.0.3] - 2024-04-30
Use a threadpool for processing locations

## [0.0.2] - 2024-04-30
Fix dataframe handling

## [0.0.17] - 2024-05-01
Fix typo in env string

## [0.0.16] - 2024-05-01
Update for global/local corods mixup

## [0.0.15] - 2024-04-30
Works processing patch-wise

## [0.0.14] - 2024-04-30
Update to use relative coords when indexing into boxes

## [0.0.13] - 2024-04-30
Process boxes in a vectorized way

## [0.0.12] - 2024-04-30
Check if the box contains any median embeddings, if so go exhaustive

## [0.0.11] - 2024-04-30
Fix median emb loop and more verbose

## [0.0.10] - 2024-04-30
Remove parallelization

## [0.0.1] - 2024-04-30
Initial deploy
