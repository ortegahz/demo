import cv2
import time

import numpy as np


def plot_hpe(frame, preds, maxvals, th_pt=0.1):
    for i, (pred, maxval) in enumerate(zip(preds[0], maxvals[0])):
        color = (255, 0, 0)
        x = pred[0]
        y = pred[1]
        if maxval[0] > th_pt:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            cv2.putText(frame, f'{maxval[0]:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for pair in skeleton:
        pt1, maxval1 = preds[0][pair[0] - 1], maxvals[0][pair[0] - 1]
        x1, y1 = pt1
        pt2, maxval2 = preds[0][pair[1] - 1], maxvals[0][pair[1] - 1]
        x2, y2 = pt2
        if maxval1[0] > th_pt and maxval2[0] > th_pt:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)


def process_displayer(max_track_num, buff_len, arr_frames, dict_tracker_res, dict_recog_res):
    sort_colours = np.random.rand(max_track_num, 3) * 255

    name_window = 'frame'
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 540)

    while True:
        if len(arr_frames) >= buff_len:

            item = arr_frames[0]
            idx_frame, frame = item
            arr_frames.pop(0)

            if idx_frame in dict_recog_res.keys():
                dets, hpe_res = dict_recog_res[idx_frame]
                dict_recog_res.pop(idx_frame)

                for hpe_res_s in hpe_res:
                    preds, maxvals = hpe_res_s
                    plot_hpe(frame, preds, maxvals)

                # plot
                for d in dets:
                    bbox = d[0:4]
                    box = bbox.astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

            # print('len(dict_tracker_res) --> %d' % len(dict_tracker_res))
            if idx_frame in dict_tracker_res.keys():
                online_targets = dict_tracker_res[idx_frame]
                dict_tracker_res.pop(idx_frame)
                # plot
                for track in online_targets:
                    tlwh = track.tlwh
                    tid = track.track_id
                    tlbr = [tlwh[0], tlwh[1], tlwh[2] + tlwh[0], tlwh[3] + tlwh[1]]
                    bbox = np.concatenate((tlbr, [tid + 1])).reshape(1, -1)
                    bbox = np.squeeze(bbox)
                    box = bbox.astype(int)
                    # print('score', faces[i][4])
                    # track = mot_tracker.trackers[len(faces) - i - 1]  # reversed order
                    # box = faces[i].astype(int)
                    sort_id = box[4].astype(np.int32)
                    color_id = sort_colours[sort_id % max_track_num, :]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_id, 2)
                    info = 'tid' + ' %d' % sort_id
                    fontScale = 1.2
                    cv2.putText(frame, info,
                                (box[0], box[1] + int(fontScale * 25)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id,
                                2)
                    # info = 'age' + ' %d' % track.age
                    # fontScale = 1.2
                    # cv2.putText(frame, info,
                    #             (box[0], box[1]+int(fontScale * 25 * 2)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color_id, 2)

            cv2.putText(frame, f'{idx_frame}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 1,
                        2)
            cv2.imshow(name_window, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # time.sleep(0.01)

    cv2.destroyAllWindows()
