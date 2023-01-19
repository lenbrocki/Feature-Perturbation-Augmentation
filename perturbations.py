import random
from numba import njit
import numpy as np

@njit()
def mask_batch_rand(batch_size, channels, width, height, max_perturb, max_box, p_box):
    batch = np.ones((batch_size, channels, width, height), dtype=np.float32)
    # set the density of perturbation for this batch
    density = np.random.uniform(0, max_perturb)
    for i in range(batch_size):
        for x in range(width):
            for y in range(height):
                t = random.random()
                # check if pixel should be set to zero
                if(t <= density):
                    # check if box should be put
                    p = random.random()
                    if(p <= p_box):
                        s_box = random.randint(0, max_box)
                        if(x + s_box < width and y+s_box < height ):
                            for xs in range(s_box):
                                for ys in range(s_box):
                                    for c in range(channels):
                                        batch[i,c,x+xs,y+ys] = 0
                    else:
                        for c in range(channels):
                                batch[i,c,x,y] = 0
    return batch

# augmentation according to arXiv:1708.04896
@njit()
def mask_batch_rect(batch_size, channels, width, height, s_low, s_high, r1):
    S = width*height
    batch = np.ones((batch_size, channels, width, height), dtype=np.float32)
    for i in range(batch_size):
        We = width+1
        He = We
        # randomly choose size of area and aspect ratio until side lengths are smaller than image width and height
        while(We > width or He > height):
            ratio = random.uniform(s_low, s_high)
            Se = round(ratio*S)

            re = np.random.uniform(r1, 1/r1)
            We = round(np.sqrt(re*Se))
            He = round(np.sqrt(Se/re))

        x = random.randint(0, width-We)
        y = random.randint(0, height-He)

        for xs in range(We):
            for ys in range(He):
                for c in range(channels):
                    batch[i,c,x+xs,y+ys] = 0
    return batch