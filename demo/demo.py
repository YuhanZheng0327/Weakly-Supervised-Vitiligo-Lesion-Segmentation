import torch
import cv2
import numpy as np
from SPANet import SPANet  # model
import os


# import model for testing
def load_model(model_path, use_cuda=True):
    model = SPANet()

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    # import checkpoint
    model.load_state_dict(torch.load(model_path))
    return model


def test_model(model, test_image_path, label_colours, use_cuda=True):
    # import test image
    test_img = cv2.imread(test_image_path)
    test_data = torch.from_numpy(np.array([test_img.transpose((2, 0, 1)).astype('float32') / 255.]))

    # using GPU
    if use_cuda and torch.cuda.is_available():
        test_data = test_data.cuda()

    test_data = test_data.float()

    # evaluate model
    model.eval()

    with torch.no_grad():
        att_, output = model(test_data)

    # output
    output = output[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, 50)
    _, predicted = torch.max(output, 1)

    predicted = predicted.data.cpu().numpy()
    predicted = predicted.reshape(test_img.shape[0], test_img.shape[1])

    # show output
    im_target_rgb = np.array([label_colours[c % 50] for c in predicted])
    im_target_rgb = im_target_rgb.reshape(test_img.shape).astype(np.uint8)

    # result
    cv2.imshow("Test Output", im_target_rgb)
    cv2.waitKey(0)  # 等待用户按下任意键关闭

    # save result
    output_name = f"./output/{os.path.basename(test_image_path)}"
    cv2.imwrite(output_name, im_target_rgb)


# 使用示例
if __name__ == "__main__":
    model_save_path = './checkpoint.pth'  # model path
    test_image_path = './image/656_1.jpg'  # testing
    label_colours = np.random.randint(255, size=(50, 3))  # label colours

    # loading model
    model = load_model(model_save_path)

    # test
    test_model(model, test_image_path, label_colours)
