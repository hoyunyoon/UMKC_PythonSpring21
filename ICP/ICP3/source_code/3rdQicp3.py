import numpy as np###library

Random_Int = np.random.randint(1, 20, 20)

print("Random Integers :", Random_Int)

Rshping = Random_Int.reshape((4, 5))

print("Random Integers ater Reshaping :")

print(Rshping)

Replacing_Max = np.where(Rshping == np.amax(Rshping, axis=1, keepdims=True), 0, Rshping)

print("Random Integers ater replacing the maximum in each row by 0")
print(Replacing_Max)
