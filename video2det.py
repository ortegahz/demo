import cv2
import os

from retinaface import RetinaFace

name_window = 'frame'
path_video = '/media/manu/samsung/videos/at2021/mp4/Video1.mp4'
path_det = '/home/manu/tmp/det.txt'
path_img = '/home/manu/tmp/imgs'

model_face_detect_path =\
    '/media/manu/intel/workspace/insightface_manu_img2rec/RetinaFace/models/manu/mobilenet_v1_0_25/retina'
warmup_img_path = '/media/manu/samsung/pics/material3000_1920x1080.jpg'  # image size should be same as actual input
gpuid = 0
thresh = 0.5
scales = [1.0]
flip = False

os.system('rm %s -rvf' % path_det)
os.system('rm %s -rvf' % path_img)
os.mkdir(path_img)

if __name__ == '__main__':

    print('face detect init start ...')
    detector = RetinaFace(model_face_detect_path, 0, gpuid, 'net3')
    img = cv2.imread(warmup_img_path)
    print('face detect init done')

    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    cap = cv2.VideoCapture(path_video)
    if cap.isOpened() is False:
        print("Error opening video steam")

    frame_idx = 0  # begin with idx 1
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            frame_idx += 1
            path = os.path.join(path_img, '%06d.jpg' % frame_idx)
            cv2.imwrite(path, frame)
            faces, landmarks = detector.detect(frame, thresh, scales=scales, do_flip=flip)
            info = 'idx %d' % frame_idx
            fontScale = 1.2
            cv2.putText(frame, info, (0, int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
            if faces is not None:
                print('find', faces.shape[0], 'faces')

            for i in range(faces.shape[0]):
                face = faces[i]
                line_ = '%d,-1,%f,%f,%f,%f,%f,-1,-1,-1\n' %\
                        (frame_idx, face[0], face[1], face[2]-face[0], face[3]-face[1], face[4])
                with open(path_det, 'a+') as f:
                    f.write(line_)

                # print('score', faces[i][4])
                box = faces[i].astype(int)
                # color = (255,0,0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            cv2.imshow(name_window, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
