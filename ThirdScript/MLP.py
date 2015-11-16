import numpy as np;

sigmoid = lambda x : 1.0/(1.0 + np.exp(-x));
sigmoid_derv = lambda x : x * (1-x);

weights = [];
def add_layers(n_din, n_dout):
	weights.append(np.random.normal(0, 1.0, (n_din, n_dout)));

hiddens = [];
def infer(din, active_function=sigmoid):
	del hiddens[:];
	output = np.array(din);
	output = output.reshape(1, output.shape[0]);
	hiddens.append(output);
	for layer in weights:
		hidden = np.dot(output, layer);
		output = np.array(map(active_function, hidden)).reshape(1, hidden.shape[1]);
		hiddens.append(output);
	return output.reshape(output.shape[1]);

def train_once(din, dout, alpha=1.0, active_function_derv=sigmoid_derv):
	network_out = infer(din);
	derv = network_out - dout;
	for layer in xrange(len(weights), 0, -1):
		derv *= np.array(map(active_function_derv, hiddens[layer].reshape(hiddens[layer].shape[1])));
		derv = derv.reshape(1, derv.shape[0]);
		weights[layer-1] -= alpha * np.dot(hiddens[layer-1].reshape(hiddens[layer-1].shape[1], 1), derv);
		derv = np.dot(derv, weights[layer-1].T);
		derv = derv.reshape(derv.shape[1]);

def init_layers(layers):
	for i in xrange(len(layers)-1):
		add_layers(layers[i], layers[i+1]);

def train(din, dout, epos, alpha=1.0):
	for i_epos in xrange(epos):
		for item_in, item_out in zip(din, dout):
			train_once(item_in, item_out, alpha);
