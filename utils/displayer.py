import cv2
import time

import numpy as np


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
                dets = dict_recog_res[idx_frame]
                dict_recog_res.pop(idx_frame)
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
