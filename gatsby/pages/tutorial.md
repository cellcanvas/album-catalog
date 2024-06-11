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

