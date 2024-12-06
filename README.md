```grovers-algorithm.ipynb``` contains the IBM implementation of Grover's Algorithm. Run the individual cells cronologically to observe results of each step in real time.

```grover-simulator.py``` contains Siumlated algorithm containing noise functions. It can be tested by the following command:

```
python grover-simulator.py --marked-states 010,101 --noise-model deploarizing
```

The number of marked states and number of qubits can be changed as per requirement and noise model can be picked from depolarizing, phase-flip and bit-flip