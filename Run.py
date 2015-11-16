import numpy as np;
import ThirdScript.MLP as MLP;

dataset = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]);
MLP.init_layers([2, 10, 10, 10, 10, 1]);
MLP.train(dataset[:, 0:2], dataset[:, 2:3], 2000, 2.0);

for item in dataset:
	print MLP.infer(item[0:2]);