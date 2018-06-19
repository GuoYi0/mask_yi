import colorsys
import numpy as np
def generate_colors(N, bright=True):
    """
    :param N: 生成的颜色总数
    :param bright:
    :return: 一个列表，列表的每个元素都是一个三元组，RGB
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    :param image: [gao,kuan, 通道数]
    :param mask: 一个rank为2的bool型数组，[高，宽]，与image有相同的高宽，True表示这里有instance
    :param color:
    :param alpha:
    :return:
    """
    for c in range(3):
        image[:, :, c] = np.where(mask,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def draw_box(image, box, color):
    """

    :param image:
    :param box:  一个包含四个元素的向量[x1, y1, x2, y2]， 像素坐标
    :param color:
    :return:
    """
    x1, y1, x2, y2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image

def apply_box_mask(image, box, mask, ids, num_class):
    """

    :param image: 输入图片，[高，宽，通道数]
    :param box: [num, (x1, y1, x2, y2)]
    :param mask: [num, (gao, kuan)]
    :param num_class: 类别总数
    :param ids: [num] 类别号
    :return:
    """
    num = np.shape(box)[0]
    colors = generate_colors(num_class)

    min_id = 1000
    max_id = 0
    for i in range(num):
        if ids[i] >= 0:
            if ids[i] > max_id:
                max_id = ids[i]
            if ids[i] < min_id:
                min_id = ids[i]
        try:
            color = colors[ids[i]]
        except IndexError:
            print("the length of colors is {}, but the index is {}".format(len(colors), ids[i]))
        image = apply_mask(image, mask[i], color)
        image = draw_box(image, box[i], color)

    print("The min id: ",  min_id, max_id)
    return image
