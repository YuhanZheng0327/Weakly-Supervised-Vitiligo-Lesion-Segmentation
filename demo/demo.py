import torch
import cv2
import numpy as np
from SPANet import SPANet  # 你的模型定义
import os


# 加载模型并进行测试
def load_model(model_path, use_cuda=True):
    # 初始化模型
    model = SPANet()

    # 如果有GPU，加载到GPU
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path))
    return model


def test_model(model, test_image_path, label_colours, use_cuda=True):
    # 读取测试图像
    test_img = cv2.imread(test_image_path)
    test_data = torch.from_numpy(np.array([test_img.transpose((2, 0, 1)).astype('float32') / 255.]))

    # 如果使用GPU，将数据传输到GPU
    if use_cuda and torch.cuda.is_available():
        test_data = test_data.cuda()

    test_data = test_data.float()

    # 将模型设为评估模式
    model.eval()

    with torch.no_grad():  # 在推理时禁用梯度计算
        att_, output, _, _, _, _ = model(test_data)

    # 获取输出结果
    output = output[0]  # 只取第一个batch
    output = output.permute(1, 2, 0).contiguous().view(-1, 50)  # HxWxC -> N, C
    _, predicted = torch.max(output, 1)

    predicted = predicted.data.cpu().numpy()
    predicted = predicted.reshape(test_img.shape[0], test_img.shape[1])

    # 显示/保存结果
    im_target_rgb = np.array([label_colours[c % 50] for c in predicted])
    im_target_rgb = im_target_rgb.reshape(test_img.shape).astype(np.uint8)

    # 显示测试结果
    cv2.imshow("Test Output", im_target_rgb)
    cv2.waitKey(0)  # 等待用户按下任意键关闭

    # 保存结果
    output_name = f"./output/{os.path.basename(test_image_path)}"
    cv2.imwrite(output_name, im_target_rgb)


# 使用示例
if __name__ == "__main__":
    model_save_path = './checkpoint.pth'  # 模型保存路径
    test_image_path = './image/656_1.jpg'  # 测试图像路径
    label_colours = np.random.randint(255, size=(50, 3))  # 标签颜色（假设最多有50个标签）

    # 加载模型
    model = load_model(model_save_path)

    # 进行测试
    test_model(model, test_image_path, label_colours)
