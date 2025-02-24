The dataset for the buoyancy-driven flow is downloaded from the PDE-Arena
link: https://huggingface.co/datasets/pdearena/NavierStokes-2D/tree/main

We then stacked all the samples into a single npy file with the following shape:
dataset: [BS, nt, nx, ny, nf] 

train dataset consists of 2080 samples
val   dataset consists of 260  samples
test  dataset consists of 260  samples

nf=3, representing the concentrationn field (d) and the velocities (vx, vy)

Therefore, the MEAN, STD, MIN, MAX was computed for each field separately
