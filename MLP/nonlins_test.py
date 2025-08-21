# Plot decision boundaries for non linear datasets
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from neuralnet import MultiLayerPerceptron

#Function block
def loadData(n_rows, dataset):
    if dataset == "circles":
        X, y = make_circles(n_samples=n_rows, noise=0.1, factor=0.3,random_state=42)
        return X,y

    if dataset == "moons":
        X, y = make_moons(n_samples=n_rows, noise=0.1, random_state=42)
        return X,y
    
    if dataset=="xor":
        X = np.random.uniform(low=-1, high=1, size=(n_rows,2))
        y = np.bitwise_xor(np.sign(X[:,0]).astype(int),np.sign(X[:,1]).astype(int))

        y = np.array([0 if i<0 else 1 for i in y])
        return X, y

    a=0
    b=1
    theta_max = 4*np.pi
    if dataset=="spiral":

        theta = np.linspace(0, theta_max, n_rows) + np.random.uniform(low=0, high=2, size=(n_rows))
        # Calculate r using the polar equation r = a + b * theta
        r = a + b * theta
        # Convert polar coordinates to Cartesian coordinates
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)

        x2 = r * np.cos(theta+180)
        y2 = r * np.sin(theta+180)
        #Make it classification dataset
        x_new = np.hstack((x1,x2))
        y_new = np.hstack((y1,y2))

        X = np.transpose(np.vstack((x_new,y_new)))
        y = np.hstack((np.ones((x1.shape[0])), np.zeros((x2.shape[0])) ))
        return X,y

    if dataset=="sunflower":

        theta = np.linspace(50, theta_max, n_rows)
        # Calculate r using the polar equation r = a + b * theta
        r = a + b * theta
        # Convert polar coordinates to Cartesian coordinates
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)

        x2 = r * np.cos(theta+180)
        y2 = r * np.sin(theta+180)
        #Make it classification dataset
        x_new = np.hstack((x1,x2))
        y_new = np.hstack((y1,y2))

        X = np.transpose(np.vstack((x_new,y_new)))
        y = np.hstack((np.ones((x1.shape[0])), np.zeros((x2.shape[0])) ))
        return X,y

def plotter(x, y, cost):
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(np.arange(cost.shape[0]), cost)
    ax2.scatter(x[:,0], x[:,1], c=y)


    # Create mesh grid for decision boundary
    x_min, x_max = xtrain[:, 0].min() - 1, xtrain[:, 0].max() + 1
    y_min, y_max = xtrain[:, 1].min() - 1, xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict over mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax2.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)
    ax2.contour(xx, yy, Z, colors='k', levels=[0, 1], 
                linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    plt.suptitle(f"Accuracy {round(acc,4)*100}%\nlr {lr}, Epochs {epochs}")


#Main block
if __name__=="__main__":
    X,y = loadData(n_rows=500, dataset="sunflower")

    #Split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y, train_size=0.7)
    print(xtrain.shape, ytest.shape)

    #Training
    model = MultiLayerPerceptron()
    
    model.add(16,inputs_dim=2)
    model.add(16)
    model.add(16)
    model.add(16)
    model.add(1, loss_function="binary_crossentropy")

    lr = 0.1
    epochs = 10000

    model.fit(xtrain, ytrain, learning_rate=lr, epochs=epochs)


    #Prediction
    y_pred = model.predict(xtest)
    y_pred = np.where(y_pred>=0.5, 1, 0)

    #Accuracy
    acc = accuracy_score(ytest, y_pred)
    print("\nAcccuracy",round(acc,4))

    #Plot
    plotter(xtrain, ytrain, model.cost)
    plt.show()