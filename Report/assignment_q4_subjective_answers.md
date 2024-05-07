# assignment_q4_subjective_answers

The function `GenerateData` generates data with discrete or real outputs and features as required. 

The function `TimeIt` returns a `DataFrame` which tells us the average time required for fitting and prediction of the tree in all 4 cases (as created by `GenerateData`). In our case, we get this output for datasets with only 10 samples.  

```
	 Disc. In. ?  Disc. Out. ?  time to fit  std for fit  time to predict  std for pred.  acc./RMSE  std for acc./RMSE
0        False         False     1.336531     0.758510         0.026349       0.027490   0.712413           0.439974
1        False          True     2.449529     0.441296         0.027903       0.007575   1.000000           0.000000
2         True         False     0.080193     0.038492         0.017119       0.012346   0.665139           0.244510
3         True          True     0.386936     0.047465         0.022461       0.005236   0.890000           0.053852
```

Trees created while calculating these averages look like this : 

Case 0 (Real Input, Real Output)

<img src=assignment_q4_subjective_answers/Case_0.png width=50%>

Case 1 (Real Input, Disc. Output)

<img src=assignment_q4_subjective_answers/Case_1.png width=50%>

Case 2 (Disc. Input, Real Output)

<img src=assignment_q4_subjective_answers/Case_2.png width=50%>

Case 3 (Disc. Input, Disc. Output)

<img src=assignment_q4_subjective_answers/Case_3.png width=50%>

The function `plot_time` plots these graphs which tell us about the complexity of the algorithm.

<p position=relative>
<img src=assignment_q4_subjective_answers/Untitled.png width=30%>
<img src=assignment_q4_subjective_answers/Untitled%201.png width=30%>
<img src=assignment_q4_subjective_answers/Untitled%202.png width=30%>
</p>

For the third plot, the upper two graphs are for the accuracy (for discrete output) while the lower two are for RMSE (for real output).

As visible, the time required to predict grows linearly with log(N), and the time required to fit grows exponentially in log(N) and thus it’s polynomial in N, where N is the size of the data-set. This is in agreement with the theoritical complexity of prediction and fitting.  
