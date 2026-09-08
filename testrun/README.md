## Gradient Boosting Decision Tree + Genetic Algorithm for finding tile sizes

The goal of this directoty is to implement a workflow as follows: (based on TVM Ansor)

<img width="755" height="1405" alt="Image" src="https://github.com/user-attachments/assets/5574ca21-6610-45ff-90ec-1e0638afa567" />

All the process is done via the `run_loop.py` script.

1. Generates a sets of random tile sizes using `generate_configs.py`, from there it takes the top configurations (sorted by bigger tiles firts)
2. Then runs those on the simulator
3. With the data from step 2 it trains the gradient boost decision tree
4. Then the Genetic Algorithm (GA) runs to find new configs 
5. Those configs are then measured by the gradient boost decision tree and the top configs are measured on the simulator
6. Train new tree with combined data
7. Loop


Now let's dive into a bit more of detail.

### Hardware Constrains 

All of the workflow takes into account hardware constrains to only generate valid configurations, these are taken from previosly exisiting code and are:

*  M % m == 0  (m must exactly divide M — no remainder tiles)
*  N % n == 0  (n must exactly divide N — no remainder tiles)
*  n is a multiple of 8  (SSR unroll-and-jam factor)
*  K % k == 0  (k must exactly divide K — no remainder tiles)
*  k >= 8 if K >= 8, else k >= 3  (HW loop prologue/epilogue minimum)
*  2 * (m\*k + n\*k + m\*n) * 8 <= L1_BYTES  (double-buffered L1 fit)
*  max(m\*k, n\*k, m\*n) * 8 <= BANK_BYTES   (8-bank scratchpad layout)

Or if `allow-boundary` is enabled (which in practise we always enable)

* m can be any value >= 8
* n is still a multiple of 8; if N % n != 0 the N-remainder must also be a multiple of 8
* k can be any value >= 8; if K % k != 0 the K-remainder must be >= 3
* L1 and 8-bank constraints applied to the main tile as before 


### Taking biggest tiles from random sampling

As said in point 1, we generate a set of random configs (`--init-pool` parameter) which end up sorted by space taken on the CSV files, then from those we take the top (`--init-configs` parameters). We do these beceause if we just took random tiles at the start it is highly like that all of them are just small and really slow and will time out leaving us without any valuable data. 

### Timeout mechanism

A timeout mechanism in time (seconds) was implemented such that if a configuration in the simulator takes more than that time it times out. There is also a tiemout mechanism in cycles that was already implemented, this is similar to that.

Choosing a good value for this timeout is important as bad tiles can take extremely long to run but this value is determined by the GEMM size itself for example for 128x128x128 we found that 45 mins (2700) is a good value. This in practise means that each iteration is 45 minutes long, with this we can also calculate how long the search is going to take.

### Configurations database (blocklist)

To avoid testing the a configuration more than once while the loop runs a file named `tested-configs.txt` will be created this containst all of the already tested configurations and is passed to the GA so it does not generate them again. This not only avoids generating again a good tile size but also blocks the tile sizes that have produced a timeout. 

### How many simulations to run (batch size)

If using `TRACE_DMA_ONLY` the log sizes should not be a major problem, still you may need ~100GB of free space (depending on how many experiments you will run) but is much less than without `TRACE_DMA_ONLY`. So the best option for batch_size is to set it to the number of cores as one simulation will only take one core.

### Features used

The features we use to train the gradient boost decision tree are the same features that the myrtle cost model uses, this makes it a fair comparison between two approaches. 

### Number of iterations / Tile size found

The number of iterations is a parameter controlled with `--iterations`. Depending on the GEMM size you may need more or less iterations between 10 and 15 is usually a good number.

### Possible Improvements

Currently there is a set of things that are worth exploring to see if the provide any kind of improvement.

* **Use gvsoc instead of verilator for a very fast search** 
* Check the parameters of the Gradient Boost Decision Tree and GA as they were chosen without too much thinking
* Instead of using a random configs for the first iteration start with already good tile sizes that come from the other cost model
* Use more/other features

