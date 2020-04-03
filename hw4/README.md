Author: Alex Wong, Rahul Suresh, Shawn Lin

# Introduction
Alpha-Beta (Kalman Filter) to detect/track bats

# Before you start
Need to download BatImages, Localization and Segmentation data from Google Drive. You can download it manually:
 - https://drive.google.com/file/d/1cA9Ov-rEnIXNBAlHQddeBuL2jIhy29Xz/view?usp=sharing (bat)
 - https://drive.google.com/file/d/1RW6NuSvh11ia0BkonseHrJthPLCijERb/view?usp=sharing (bat)
 - https://drive.google.com/file/d/1ZVMDgPTWCN5mSkv1Ah9ssq20REFC2sVh/view?usp=sharing (bat)
 - https://drive.google.com/file/d/1IItlPF7Nj9GsZrAlqhHRs6e3aon732hu/view?usp=sharing (cell)

And, install all the necessary libraries (shown on first few lines of code.py)

# Run code:
 - python3 <select a py file below>.py
 - To go to next frame, hit 'q'.

# Files
bat_data_loader.py: run alpha beta filter on bat dataset
cell_data_loader.py: run alpha beta filter on cell dataset
alpha_beta_filter.py: run alpha beta filter on original dataset localization file provided.
