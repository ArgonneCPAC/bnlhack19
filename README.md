# chopperhack19
Source code for the BNL GPU Hackathon

## Data for Testing

- https://portal.nersc.gov/project/hacc/aphearin/BNLHack19/
- https://drive.google.com/file/d/1JZn3fbBELazpnrlw-0WiGzBkNZaFlT_7/view?usp=sharing

## Code organization notes

* The code in the `sf_history` subpackage maps galaxy properties onto halos. For example, the `mean_sfr` function maps SFR onto halos across time; this function needs to be integrated to map stellar mass onto halos. 
* The code in the `mock_obs` subpackage computes summary statistics of the galaxy population. The `numba_gw_hist` function calculates a Gaussian-weighted histogram, the kernel underlying a stellar mass function.  
