import cv2


def draw_boxes(img, boxes, scale=None, mode=0, color=(255, 0, 255)):
    if mode == 1:
        for box in boxes:
            cv2.line(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, 1)
            cv2.line(img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color, 1)
            cv2.line(img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color, 1)
            cv2.line(img, (box[0][0], box[0][1]), (box[3][0], box[3][1]), color, 1)

    elif mode == 0:
        for box in boxes:
            # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            #     continue
            # if box[8] >= 0.9:
            #     color = (0, 255, 0)
            # elif box[8] >= 0.8:
            #     color = (255, 0, 0)

            # if not np.where((box < 512) & (box > 0))[0].shape[0] == 4:
            #     continue
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), color, 1)
            cv2.line(img, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            cv2.line(img, (int(box[2]), int(box[3])), (int(box[0]), int(box[3])), color, 1)
            cv2.line(img, (int(box[0]), int(box[3])), (int(box[0]), int(box[1])), color, 1)

        # min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        # min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        # max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        # max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        #
        # tm = box[4]
        # box[4] = box[6]
        # box[6] = tm
        #
        # tm = box[5]
        # box[5] = box[7]
        # box[7] = tm
        #
        # line = functools.reduce(lambda x, y: '{},{}'.format(x, y), box)
        # f.write(line)
        # f.write('\n')

        # line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
        # f.write(line)
    return img

    # img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
