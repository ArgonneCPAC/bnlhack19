# chopperhack19
Source code for the BNL GPU Hackathon

## Data for Testing

- https://portal.nersc.gov/project/hacc/aphearin/BNLHack19/
- https://drive.google.com/file/d/1JZn3fbBELazpnrlw-0WiGzBkNZaFlT_7/view?usp=sharing

## Code organization notes

* The code in the `sf_history` subpackage maps galaxy properties onto halos. For example, the `mean_sfr` function maps SFR onto halos across time; this function needs to be integrated to map stellar mass onto halos. 
* The code in the `mock_obs` subpackage computes summary statistics of the galaxy population. The `numba_gw_hist` function calculates a Gaussian-weighted histogram, the kernel underlying a stellar mass function.  

## Getting Setup at BNL

1. Install miniconda. 
    
   I've been told it is better to keep it out of your home area, but that might just be my account. I made a directory on the 
   global scratch area, symlinked it to my home area, and then installed there.

2. Use `conda init` to setup your shell if you did not install it while doing miniconda.

3. Add `module load "cuda/9.0"` to your `~/.bashrc".

4. Install all of the software you need in your base conda env.

5. Get a node `srun --pty -A hackathon -p long -t 30:00 /bin/bash`
