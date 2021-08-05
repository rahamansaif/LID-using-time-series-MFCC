
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np 
import sys
tf.disable_v2_behavior()

def forward_hiddenLayers(W, b, a):
    Z = tf.matmul(W,a) + b;
    A = tf.nn.relu(Z);
    return A;


#as the last unit can be both softmax(in case of a multiclass classifier) as well as a sigmoid it is left for later spefication by the user
def forward_Propagation(X, parameters, Nlayers):
    #nlayers is the number of hidden layers + 1
    #getting parameters (1 all the way to L) from the dictionary and completing forward propagation
    A = X;
    for l in range(1,Nlayers):
        Wi = parameters['W'+str(l)];
        bi = parameters['b'+str(l)];
        A = forward_hiddenLayers(Wi, bi, A);
    #the last unit is not passed through the sigmoid here, as we would need a softmax classifier
    Zout = tf.matmul((parameters['W'+str(Nlayers)]), A) + parameters['b'+str(Nlayers)];
    return Zout;        

def initialize_parameters(Nlayers, dimensions):
    #L is the number of hidden layers + 1
    parameters = {};
    initializer = tf2.initializers.GlorotUniform(seed=2);
    for l in range(1, Nlayers+1):
        parameters['W'+str(l)] = tf2.Variable(initializer(shape=(dimensions[l], dimensions[l-1])));
        parameters['b'+str(l)] = tf2.Variable(initializer(shape=(dimensions[l], 1)));
    return parameters;

def Cost_function(Zout, Y):
    #Z is the output of the linear layer(last layer the softmax is yet to be applied)
    #logits indicate the output of the classifier (as our logits are column vectors here, and tensorflow wants it to be a row vector)
    epsilon = 1E-8;
    logits = tf.transpose(Zout);
    logits = tf.nn.softmax(logits+epsilon);
    labels = tf.transpose(Y);
    #finds the softmax loss (i.e. sum (yi.log(yhati)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels));
    return cost;

def create_placeHolders(n_x, n_y):    
    X = tf.placeholder(tf.float32, [n_x, None], name='X');
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
    return X, Y

def shuffleTrainSet_And_GET_mini_batches(X_train, Y_train, mini_batch_size, seed):
    #we have to shuffle X_train and Y_train and arrange them into minibatch matrices
    np.random.seed(seed);
    m = X_train.shape[1];      
    mini_batches = [];
    #m is the number of training example, the command returns a list of the permutations of all the whole numnbers from 0 to m-1
    order_permute = list(np.random.permutation(m)); 
    #now permute X_train and Y_train according to the order permute
    shuffled_X = X_train[:, order_permute].reshape((X_train.shape[0], m));
    shuffled_Y = Y_train[:, order_permute].reshape((Y_train.shape[0], m));
    num_complete_minibatches = int(m / mini_batch_size);  #number of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size];
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size];
        mini_batch = (mini_batch_X, mini_batch_Y);
        mini_batches.append(mini_batch);
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
    return mini_batches;

def SaveParameters(parameters, filename, Nlayers):
    for l in range(1, Nlayers+1):
        np.save(filename+'W'+str(l)+'.npy', parameters['W'+str(l)]);
        np.save(filename+'b'+str(l)+'.npy', parameters['b'+str(l)]);

def train_model(Nlayers, dimensions, X_train, Y_train, learning_rate=0.0001, total_epochs = 2000, mini_batch_size=64, printCost = False):
    tf.reset_default_graph();   #this will help rerun the model, instead of starting from the previous run value
    (n_x, m) = X_train.shape;                    #number of features and number of training examples     
    n_y = Y_train.shape[0];
    X, Y = create_placeHolders(n_x, n_y);
    costs = [];                                   # To keep track of the cost if required to plot
    parameters = initialize_parameters(Nlayers, dimensions);
    
    #setting up the graph
    Zout = forward_Propagation(X, parameters, Nlayers);
    cost = Cost_function(Zout, Y);
    #complete graph set up
    ########################################Training the model#################################################################
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost);
    init = tf.global_variables_initializer();
    with tf.Session () as sess:
        sess.run(init);
        seed  = 1;
        for epoch in range(total_epochs):
            #here you make all the minivbatches....i.e. perform mini-batch gradient descent 
            epoch_cost = 0.0;                           
            num_minibatches = int(m / mini_batch_size);  #number of minibatches of size minibatch_size in the train set
            seed = seed + 1;                            #changing the seed in every epoch
            mini_batches = shuffleTrainSet_And_GET_mini_batches(X_train, Y_train, mini_batch_size, seed);
            for mini_batch in mini_batches:
                (mini_X, mini_Y) = mini_batch;
                batch_size_k = mini_X.shape[1];
                _, mini_cost = sess.run([optimizer, cost], feed_dict={X:mini_X, Y:mini_Y});
                epoch_cost = epoch_cost + mini_cost/batch_size_k;
            #printing cost after every 100 epochs
            if(epoch%200 == 0):
                parapresent = sess.run(parameters);
                SaveParameters(parapresent, "models/", Nlayers);
                #saver.save(sess, "models/parameters.ckpt");
            if(printCost == True and epoch % 500 == 0):
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost));
            if(epoch % 5 == 0):
                costs.append(epoch_cost);        #to plot the cost vs the epoch graph sampled every 5 epochs
        #this will display the cost 
        plt.plot(np.squeeze(costs));
        plt.ylabel('Cost');
        plt.xlabel('Epochs (every 5 epoch)');
        plt.title("Learning rate = " + str(learning_rate));
        plt.savefig("CostFuntionwithIter.png");
        plt.show();
        
        parapresent = sess.run(parameters);
        SaveParameters(parapresent, "models/", Nlayers);
        print("All parameters have been learnt and saved");
        ####################################end of training the model####################################################

def printResults(Y_hat, Y, ClassNames, toPrint = False):
    Y_hat_generic, Y_generic = create_placeHolders(Y.shape[0], Y.shape[0]);
    Y_labels = tf.argmax(Y_generic);
    Y_hat_labels = tf.argmax(Y_hat_generic);            #calculated column wise
    correct_prediction = tf.equal(Y_hat_labels, Y_labels); 
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))*100; 
    conf_matrix = tf.math.confusion_matrix(Y_hat_labels, Y_labels, len(ClassNames));      ##A[yhat][y] += 1 if y is predicted as yhat
    #checks if the index of the predicted and actual label is same #obtaining the percentage accuracy
    with tf.Session() as sess:
        #if you want to print clip wise observations
        if(toPrint == True):
            Y_names = [ClassNames[i] for i in sess.run(Y_labels, feed_dict = {Y_generic: Y})];
            Y_hat_names = [ClassNames[i] for i in sess.run(Y_hat_labels, feed_dict = {Y_hat_generic: Y_hat})];
            print("Actual  Predicted\n")
            for i in range(len(Y_names)):
                print(Y_names[i],"  ", Y_hat_names[i])
        
        print("\nAccuracy: ", accuracy.eval({Y_hat_generic: Y_hat, Y_generic: Y}));
        print("\nConfusion matrix: ");
        print(conf_matrix.eval({Y_hat_generic: Y_hat, Y_generic: Y}));



#Zout is the computation graph for forward propagation
def test_model(X_train, Y_train, X_test, Y_test, ClassNames, parameters, NLayers, filename, writeToFile = False):
    epsilon = 1E-8;
    X, _  = create_placeHolders(X_train.shape[0], Y_train.shape[0]);     #number of features in X and Y
    
    Zout = forward_Propagation(X, parameters, NLayers);  #since softmax is monotonic, this and the above statement are equiv.
    
    Zoutput, _ = create_placeHolders(Y_train.shape[0], Y_train.shape[0]);
    Yhat_calc = tf.nn.softmax((Zoutput+epsilon), axis=0);
    
    
    init = tf.global_variables_initializer();
    with tf.Session() as sess:
        sess.run(init);
        Zout_train = sess.run(Zout, feed_dict = {X: X_train});
        Zout_test = sess.run(Zout, feed_dict = {X: X_test});
        
        Y_hat_train = sess.run(Yhat_calc, feed_dict = {Zoutput: Zout_train});
        Y_hat_test = sess.run(Yhat_calc, feed_dict = {Zoutput: Zout_test});
        
        original_out = sys.stdout;              #original state
        f = "";
        if(writeToFile == True):
            f = open(filename, "w");
            sys.stdout = f;
            
        print("Order of ClassNames", ClassNames);
        print("Results on the train set \n");
        printResults(Y_hat_train, Y_train, ClassNames);
        print("Results on the test set \n");
        printResults(Y_hat_test, Y_test, ClassNames, True);
        if(writeToFile == True):
            f.close();
        sys.stdout = original_out;
def convert_to_one_hot_matrix(labels, Nlabels):
    M = tf.constant(Nlabels, name='C');
    one_hot_matrix = tf.one_hot(labels-1, M, axis=0);
    # Create the session (approx. 1 line)
    with tf.Session() as sess:
        one_hot_matrix = sess.run(one_hot_matrix);
    return one_hot_matrix;

def loadParameters(filename, Nlayers):
    parameters = {};
    for l in range(1, Nlayers+1):
        parameters['W'+str(l)] = np.load(filename+'W'+str(l)+'.npy');
        parameters['b'+str(l)] = np.load(filename+'b'+str(l)+'.npy');
    return parameters;


#use for training and testing the NN
def Train_Tester(X_train_loc, Y_train_loc, X_Dev_loc, Y_Dev_loc):
    #print("Here")
    X_train = (np.loadtxt(X_train_loc, delimiter=',', skiprows=1)[:, 1:]).T;
    Y_train = (np.loadtxt(Y_train_loc, delimiter=',', skiprows=1)[:, 2]).T;
    Y_train = convert_to_one_hot_matrix(Y_train,6);
    X_Dev = (np.loadtxt(X_Dev_loc, delimiter=',', skiprows=1)[:, 1:]).T;
    Y_Dev = (np.loadtxt(Y_Dev_loc, delimiter=',', skiprows=1)[:, 2]).T
    Y_Dev = convert_to_one_hot_matrix(Y_Dev,6);
    filename = "OutputPrediction.txt";
    writeToFile = True;
    init = tf.global_variables_initializer();
    with tf.Session() as sess:
        sess.run(init);
    nfeatures = X_train.shape[0];
    nClasses = 6;
    ClassNames = ['Bengali', 'English', 'Hindi', 'Marathi', 'Tamil', 'Telugu'];
    #print(X_train.shape, "", Y_train.shape, "", X_Dev.shape, "", Y_Dev.shape);
    train_model(2, [nfeatures, 500, nClasses], X_train, Y_train,  0.01, 1000, 16, True);
    parameters = loadParameters("models/", 2);
    test_model(X_train, Y_train, X_Dev, Y_Dev, ClassNames, parameters, 2, filename, writeToFile);


