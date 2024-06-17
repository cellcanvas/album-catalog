---
layout: page
title: Tutorial
permalink: /tutorial/
---

# Overview


We will work on this dataset:
https://cryoetdataportal.czscience.com/datasets/10301


## Importing the dataset

Import from the data portal into copick:

[`copick/project_from_dataportal`](https://album.cellcanvas.org/copick/project_from_dataportal/0.1.10)


This will fetch all of the data you need for the rest of this
tutorial. 

The following command will take some time, because it downloads
the zarr files for the tomograms. There are 18 runs in this
dataset. On residential internet each run takes 2.5 minutes to
download. On an HPC, each run takes ~45 seconds.

```
album run copick:project_from_dataportal:0.1.10 --dataset_id 10301 --copick_config_path ~/cellcanvas_tutorial/copick.json --overlay_root ~/cellcanvas_tutorial/overlay --static_root ~/cellcanvas_tutorial/static
```

When the command is complete you should be able to see the files you
downloaded like this:


```
$ ls ~/cellcanvas_tutorial/static/ExperimentRuns/
01122021_BrnoKrios_arctis_lam1_pos4          08042022_BrnoKrios_Arctis_grid5_gistest_Position_4  17072022_BrnoKrios_Arctis_p3ar_grid_Position_101  27042022_BrnoKrios_Arctis_grid9_hGIS_Position_33
01122021_BrnoKrios_arctis_lam1_pos5          12052022_BrnoKrios_Arctis_grid_newGISc_Position_38  17072022_BrnoKrios_Arctis_p3ar_grid_Position_35   27042022_BrnoKrios_Arctis_grid9_hGIS_Position_44
01122021_BrnoKrios_arctis_lam2_pos13         14042022_BrnoKrios_Arctis_grid5_Position_1          17072022_BrnoKrios_Arctis_p3ar_grid_Position_68   27042022_BrnoKrios_Arctis_grid9_hGIS_Position_7
01122021_BrnoKrios_arctis_lam3_pos27         15042022_BrnoKrios_Arctis_grid9_Position_32         17072022_BrnoKrios_Arctis_p3ar_grid_Position_76
06042022_BrnoKrios_Arctis_grid7_Position_29
15042022_BrnoKrios_Arctis_grid9_Position_65
27042022_BrnoKrios_Arctis_grid9_hGIS_Position_13
```

## Creating features/embeddings

Now let's create a set of features to use with this solution: [`copick/generate-skimage-features`](https://album.cellcanvas.org/copick/generate-skimage-features/0.1.13)

```
album run copick:generate-skimage-features:0.1.13 --copick_config_path
~/czii/cellcanvas_tutorial/copick.json --run_name
01122021_BrnoKrios_arctis_lam1_pos4 --voxel_spacing 7.84 --tomo_type
albumImportFromCryoETDataPortal --feature_type skimage001
```

This will create and populate a zarr file that contains features
generated with scikit-image's `multiscale_basic_features` method. That
zarr will live here:

`~/cellcanvas_tutorial/overlay/ExperimentRuns/01122021_BrnoKrios_arctis_lam1_pos4/VoxelSpacing7.840/albumImportFromCryoETDataPortal_skimage001_features.zarr/`

### TODO: add CellCanvas embeddings

This requires a pretrained CellCanvas model to be posted online.

## Creating your first annotations

Use existing picks to create painting annotations for CellCanvas.


## Inspecting the dataset


Inspecting in napari

Inspecting in cellcanvas

Inspecting in neuroglancer

