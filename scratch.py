from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# 读取文件
# pil_img = Image.open('1.jpg',)
# 读取cv2文件
img = cv2.imread('/media/manu/samsung/pics/material3000_1920x1080.jpg')
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# pil_img.show()
# 生成画笔
draw = ImageDraw.Draw(pil_img)
# 第一个参数是字体文件的路径，第二个是字体大小
font = ImageFont.truetype('SimHei.ttf', 30, encoding='utf-8')
# 第一个参数是文字的起始坐标，第二个需要输出的文字，第三个是字体颜色，第四个是字体类型
draw.text((700, 450), '黄喆', (0, 255, 255), font=font)

# PIL图片转cv2
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# 变得可以拉伸 winname 必须要一样，且设置可以拉伸在前面
cv2.namedWindow('w_img', cv2.WINDOW_NORMAL)
# 显示
cv2.imshow("w_img", img)
# 等待
cv2.waitKey(0)