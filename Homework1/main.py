import os

import numpy
import numpy as np
from PIL import Image, ImageOps

import homwork1_mnist as model

input_var = input(
    "Enter: \n create = create model\n eval = evaluate model\n tune = hyperparameter optimization\n test = test user image\n set = train model with user variables \n tensorboard= view tensorboard\n exit = exit \n command: ")
while True:
    if input_var != 'exit':
        if input_var == 'create':
            model.createModel()
        elif input_var == 'eval':
            model.eval()
        elif input_var == 'tune':
            model.tune()
        elif input_var == 'tensorboard':
            os.system("tensorboard --logdir logs")
        elif input_var == 'clear':
            os.system("rm -rf ./logs/")
        elif input_var == 'test':
            loc = input("input file location: ")
            img = 0
            try:
                img = Image.open(loc)
            except FileNotFoundError:
                print("no file found at ", loc)
                continue
            img = ImageOps.grayscale(img)
            imgArray = numpy.array(img)
            if (imgArray.shape != (28, 28)):
                print("image is wrong size")
                continue
            arr = np.expand_dims(imgArray, 0)
            model.test(arr)
        elif input_var == 'set':
            l1 = int(input("Input first dense layer: "))
            l2 = int(input("Input second dense layer: "))
            epochs = int(input("Input number of epocs: "))
            model.createModel(l1, l2, epochs)
        input_var = input(
            "Enter: \n create = create model\n eval = evaluate model\n tune = hyperparameter optimization\n test = test user image\n set = train model with user variables \n tensorboard= view tensorboard\n exit = exit \n command: ")

    else:
        break
