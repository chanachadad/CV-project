import restore_image
import models.color_image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def choose_action(kind_of_act, _image, num_res_blocks=5):
    un_image = restore_image.read_image(_image, 1)
    cv.imwrite("_original.jpg", (un_image * 255).astype(np.uint8))

    un_fix_image = un_image
    # un_fix_image = restore_image.add_gaussian_noise(un_image, 0, 0.2)
    # cv.imwrite("_dirty.jpg", (un_fix_image * 255).astype(np.uint8))

    plt.imshow(un_fix_image,  cmap='gray')  # from the imshow docstring- in case of RGB data, cmap is ignored.
    plt.show()

    # save the model:
    # model = restore_image.learn_denoising_model()
    # saved_model = model.save_weights("./denoising_model")
    model = restore_image.build_nn_model(16,16,32,5)
    saved_model = model.load_weights("./denoising_model")

    if kind_of_act == 1 or kind_of_act == 2:
        un_fix_image = restore_image.restore_image(un_fix_image, model)
        _image = "_restored.png"
        plt.imshow(un_fix_image,  cmap='gray')
        plt.show()
        cv.imwrite(_image, (un_fix_image * 255).astype(np.uint8))
        if kind_of_act == 1:
            outputFile = 'restored.png'
            cv.imwrite(outputFile, (un_fix_image * 255).astype(np.uint8))
            print('cleaned image saved as ' + outputFile)
            print('Done !!!')
            return
    if kind_of_act == 2 or kind_of_act == 3:
        models.color_image.color_the_image(_image)


choose_action(2, "imp.jpg")
