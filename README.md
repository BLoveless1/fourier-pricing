# Pricing exotic options with Fourier based methods
___

This Fourier pricing library allows the pricing of barrier and alpha-quantile options with varyng parameters under Levy and Heston models. 

### There are two distinct pricers
1. Barrier pricer
2. Alpha-quantile pricer

All pricing requests and settings are within `alpha_quantile_requests.json` and `barrier_requests.json`. A simple check is performed if only a single option type is requested:
```
# 0 - off, 1 - on
Barrier = 0
AlphaQuantile = 1
```
Change this in `main.py` if required. Alternatively the json files can be left with only a dictionary, i.e. `{}`.
