# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.7] - 2024-10-09
Another zarr group fix

## [0.2.6] - 2024-10-09
Fix segmentation zarr group

## [0.2.5] - 2024-10-09
More verbose for missing tomogram

## [0.2.4] - 2024-10-09
More debugging for missing tomogram

## [0.2.3] - 2024-08-11
Remove defaults

## [0.2.2] - 2024-07-31
Bug fix for particle radius

## [0.2.1] - 2024-07-19
Mode for processing all runs, paint in memory then write to disk

## [0.2.0] - 2024-07-19
Use particle radius and a factor, speed up painting

## [0.1.9] - 2024-05-15
Fix docstring

## [0.1.8] - 2024-05-15
Add allowlist for filtering picks by user_id

## [0.1.7] - 2024-05-14
Remove underscore from default painting segmentation name

## [0.1.6] - 2024-05-14
Skip prepicks

## [0.1.5] - 2024-05-14
Get user_id and session_id from pick_set

## [0.1.4] - 2024-05-14
Track user_id and session_id

## [0.1.3] - 2024-05-13
Display sources of pickable objects that will get painted

## [0.1.2] - 2024-05-10
Remove user_id filter for picks

## [0.1.12] - 2024-06-17
Add tomo_type as an argument

## [0.1.11] - 2024-06-17
Make voxel spacing a float

## [0.1.10] - 2024-05-28
Use copick from git

## [0.1.1] - 2024-05-10
Paint into zarr array with 1 level of indexing only

## [0.1.0] - 2024-05-10
Process per run only, specify run name

## [0.0.3] - 2024-05-09
Remove hard coded max index
