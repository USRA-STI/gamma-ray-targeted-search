
import numpy as np

from results import Results

results = Results.create(3, template_names=["soft", "norm", "hard"])

# Examples for filling results data
#results.data[] = (central_time, duration, ra, dec, template, flux_amplitude, reduced_chisq, chiplusdof, loglr, coinclr)
results.data[0] = (0.0, 0.064, 230.78080205305292, -20.266343082912595, 0, 0.931288849028193,
              0.9548338114540058, 1.1570484918296247, 7.795377147248264, 8.161539161640205)
results.data[1] = (1.0, 0.512, 179.92521315739688, -37.856411162791694, 1, 1.0965557215636013,
              0.9613880464251917, 0.9889659630100134, 73.60529823969323, 73.97499695096352)
results.data[2] = (2.0, 2.048, 189.72426327901644, -17.314776686443466, 2, 0.20927779375370187, 
              1.0952847929511258, 0.8520230378340325, 28.089802212464015, 28.459500923734296) 

results.save(".", "results.npz")

opened_results = Results.open("results.npz")
opened_results.data.sort(order='duration')
for entry in opened_results.data:
    print(entry)

