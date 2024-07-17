# SfM applications for 3D reconstruction from 2D avionics industrial inspection videos

## Description
This project consists of several changes on top of existing repositories in an effort to improve 3D reconstruction results for specific avionics industrial inspection videos. The list of the changes proposed can be found alongside the thesis at https://repository.tudelft.nl/record/uuid:5d33eed9-8939-4f1b-ba88-47e579d05cce.

## Data
The data is not available publicly.

## Environments
All environments LLFF, NeRF_PL, DSNeRF and Hierarchical-Localization work best separately, with updated requirements.txt files for all of them (following what was used in during the thesis project).

## Changes
All changes made within other codebases are marked with ```### DESCRIPTION OF CHANGE ### ``` for the start and end of the edited sections. Most testing and experimentation was done in the upper levels of code, taking part in the sfm_example.ipynb in the Hierarchical Localization folder in this repository, through the pycolmap library. 

## LLFF
LLFF was used to generate the pose files needed by many NeRF implementations that do not estimate the camera poses themselves. These are stored in 'poses_bounds.npy' files. For ease of access in case the NeRF method does not include the img2poses.py file needed, it is included in the LLFF repository. It had no other use during this project.

## DSNeRF
DSNeRF already includes the aforementioned LLFF code snippet, however it also has some changes from the original repository in the run_nerf.py file. Example config files used for this project can be found in the config folder.

## NeRF_pl
No changes were made to the main code of nerf_pl, however the extract_mesh_dsnerf.ipynb was adapted to work with the DSNeRF models as well. Color extraction for the mesh is not functional. The code used to load in a new DSNeRF model is found in utils/__init__.py and the helper methods are imported over at run_nerf_helpers_dsnerf.py. The original can be found at extract_mesh_original.ipynb. For instructions on how to use the repository, follow the README from the original author.

## Hierarchical-Localization
Most of the code in the HLOC repository is untouched as we do not use it, however inside the hloc folder the pairs_from_exhaustive.py file was changed to do sequential matching, viz_3d.py was changed slightly to introduce a noise metric for sparse reconstructions, and an example notebook of the code used to generate the results shown in the thesis paper (alongside instructions on how to use it) can be found at sfm_example.ipynb. During this project, COLMAP was setup locally inside of the Hierarchical-Localization folder for easy access.

## COLMAP
Many changes during this process were made to the source files of the COLMAP repository. This repository is not included here due to its size, and many of the  changes require some familiarity with the codebase to elaborate on. Most of the options are available through the GUI version of COLMAP, however to use it with DISK and SuperGlue (feature extraction and matching methods used in this work), COLMAP was compiled from source. Most changes were on the route of testing the necessity of different parts of the SfM pipeline (one example is turning off global bundle adjustment during the SfM process) or fixing unwanted behaviour (wrong undistortion results for SIMPLE_RADIAL camera models were fixed by changing the minimum ratio allowed for the end resulting images). Unfortunately, any change requires a rebuild of the COLMAP codebase (and pycolmap afterwards), which is not ideal.
Setting up COLMAP locally was not simple, however this github issue (https://github.com/colmap/colmap/issues/2481) and the comments by Linusnie solved most of my problems. For any additional problems, it is highly advised to search through the issues posted in the COLMAP repostiory, or open new issues.
 