import pdb

from ICA.rerank import re_ranking
import numpy as np
import os
import cv2
import pickle
import sys

sys.path.append('../')
input_path = './input'
output_pkl_path = './output_pkl'

infor_path = input_path + '/cam_infor/'
mask_loose_path = input_path + '/cam_infor/travel_time_loose_constraint.txt'
two_track_valid_pass_time_for_mask_dict = {}
with open(mask_loose_path, 'r') as f:
    for line in f:
        mask_loose_constraint = line.strip().split(' ')
        cam_out_id = int(mask_loose_constraint[0])
        cam_in_id = int(mask_loose_constraint[1])
        tth_min = int(mask_loose_constraint[2])
        tth_max = int(mask_loose_constraint[3])
        two_track_valid_pass_time_for_mask_dict[(cam_out_id, cam_in_id)] = [tth_min, tth_max]

# two_track_valid_pass_time_for_mask_dict = {(10, 11): [-10, 200], (11, 12): [-10, 100], (12, 13): [-30, 200],
#                                            (13, 14): [-250, 0], (13, 15): [-100, 50],
#                                            (15, 13): [-200, 50], (14, 13): [0, 300], (13, 12): [-200, 200],
#                                            (12, 11): [-50, 150], (11, 10): [-300, 150]}
#

mask_hard_path = input_path + '/cam_infor/travel_time_hard_constraint.txt'
two_track_valid_pass_time_dict = {}
with open(mask_hard_path, 'r') as f:
    for line in f:
        mask_hard_constraint = line.strip().split(' ')
        cam_out_id = int(mask_hard_constraint[0])
        cam_in_id = int(mask_hard_constraint[1])
        tth_min = int(mask_hard_constraint[2])
        tth_max = int(mask_hard_constraint[3])
        two_track_valid_pass_time_dict[(cam_out_id, cam_in_id)] = [tth_min, tth_max]
# loose constraint for post-process
# two_track_valid_pass_time_dict = {(10, 11): [-20, 300], (11, 12): [-20, 200], (12, 13): [-40, 200],
#                                            (13, 14): [-300, 0], (13, 15): [-100, 50],
#                                            (15, 13): [-200, 50], (14, 13): [0, 350], (13, 12): [-250, 250],
#                                            (12, 11): [-80, 200], (11, 10): [-350, 200]}

adjacent_path = input_path + '/cam_infor/adjacent_list.txt'
adjacent_list = {}
with open(adjacent_path, 'r') as f:
    for line in f:
        adjacent = line.strip().split(' ')
        cam_out_id = int(adjacent[0])
        cam_in_id = int(adjacent[1])
        out_dir = int(adjacent[2])
        in_dir = int(adjacent[3])
        adjacent_list[(cam_out_id, cam_in_id)] = [out_dir, in_dir]

# adjacent_list = {
#     (10, 11): (1, 1), (11, 10): (2, 2),
#     (11, 12): (1, 1), (12, 11): (2, 2),
#     (12, 13): (1, 1), (13, 12): (2, 2), (13, 15): (4, 2),
#     (13, 14): (1, 1), (14, 13): (2, 2),
#     (15, 13): (1, 4)
# }
keys = list(adjacent_list.keys())

from operator import itemgetter

path1 = []
path2 = []
for k in keys:
    if k[0] < k[1]:
        path1.append(k)
    else:
        path2.append(k)
path1.sort(key=itemgetter(0))
path2.sort(key=itemgetter(0), reverse=True)

time_stamps = {
    10: int(8.715 * 10),
    11: int(8.457 * 10),
    12: int(5.879 * 10),
    13: 0,
    14: int(5.042 * 10),
    15: int(8.492 * 8)
}


class MultiCameraMatching(object):
    """ This class is used to match tracklets among all different cameras.

    Attributes:
        cam_arr: camera id array
        track_arr: tracklet array
        in_dir_arr: the "in zone" in our paper
        out_dir_arr: the "out zone" in our paper
        in_time_arr: the time when a tracklet enter the "in zone"
        out_time_arr: the time when a tracklet exit the "out zone"
        feat_dict: the tracklet features
        feat_arr: it has been deprecated
    """

    def __init__(self, cam_arr, track_arr, in_dir_arr, out_dir_arr,
                 in_time_arr, out_time_arr, feat_dict,
                 topk=7, r_rate=0.5,
                 k1=12, k2=7, lambda_value=0.6,
                 alpha=1.1, long_time_t=500, short_time_t=500,
                 num_search_times=1,
                 trun_dict=None, occ_rate=1., occ_alpha=0.):
        self.cam_arr = cam_arr
        self.track_arr = track_arr
        self.in_dir_arr = in_dir_arr
        self.out_dir_arr = out_dir_arr
        self.in_time_arr = in_time_arr
        self.out_time_arr = out_time_arr

        self.feat_dict = feat_dict
        # self.feat_arr = feat_arr
        self.trun_dict = trun_dict

        ### params
        self.topk = topk
        self.r_rate = r_rate

        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

        self.alpha = alpha
        self.long_time_t = long_time_t
        self.short_time_t = short_time_t

        self.num_search_times = num_search_times

        self.occ_rate = occ_rate
        self.occ_alpha = occ_alpha

        self.global_id_arr = np.zeros_like(cam_arr) - 1

    def select_map_arr2(self, cam_id_out, cam_id_in):

        candidates_out = (self.cam_arr == cam_id_out)
        candidates_in = (self.cam_arr == cam_id_in)
        dir_map_out = np.zeros_like(candidates_out)
        dir_map_in = np.zeros_like(candidates_in)

        valid_directions = adjacent_list[(cam_id_out, cam_id_in)]
        tmp_map_out = self.out_dir_arr == valid_directions[0]
        tmp_map_in = self.in_dir_arr == valid_directions[1]

        dir_map_out |= tmp_map_out
        dir_map_in |= tmp_map_in

        candidates_out &= dir_map_out
        candidates_in &= dir_map_in
        return candidates_out, candidates_in

    def do_matching(self, cam_out_arr, cam_in_arr, track_out_arr, track_in_arr,
                    out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id, st_dim=0, en_dim=2048):
        n_out = cam_out_arr.shape[0]
        cam_out_feat_list = []
        track_out_feat_list = []
        index_out_list = []
        feat_out_list = []
        trun_out_list = []
        for i in range(n_out):
            # f_out = np.array(self.feat_dict[cam_out_arr[i]][track_out_arr[i]])[:, st_dim:en_dim]
            f_out = np.array(self.feat_dict[cam_out_arr[i]][track_out_arr[i]])[:, st_dim:en_dim]
            trun_out = np.array(self.trun_dict[cam_out_arr[i]][track_out_arr[i]])
            #             cam_out_feat_list.append(np.ones(f_out.shape[0]) * cam_out_arr[i])
            #             track_out_feat_list.append(np.ones(f_out.shape[0]) * track_out_arr[i])
            index_out_list.append(np.ones(f_out.shape[0], dtype=np.int64) * i)
            feat_out_list.append(f_out)
            trun_out_list.append(trun_out)
        #         cam_out_feat_arr = np.concatenate(cam_out_feat_list) # n
        #         track_out_feat_arr = np.concatenate(track_out_feat_arr) # n
        index_out_arr = np.concatenate(index_out_list)
        feat_out_arr = np.concatenate(feat_out_list)  # nxc
        trun_out_arr = np.concatenate(trun_out_list)  # n
        print('done for preparing feat_out_arr')

        n_in = cam_in_arr.shape[0]
        cam_in_feat_list = []
        track_in_feat_list = []
        index_in_list = []
        feat_in_list = []
        trun_in_list = []
        for j in range(n_in):
            f_in = np.array(self.feat_dict[cam_in_arr[j]][track_in_arr[j]])[:, st_dim:en_dim]
            trun_in = np.array(self.trun_dict[cam_in_arr[j]][track_in_arr[j]])

            #             cam_in_feat_list.append(np.ones(f_in.shape[0]) * cam_in_arr[j])
            #             track_in_feat_list.append(np.ones(f_in.shape[0]) * track_in_arr[j])
            index_in_list.append(np.ones(f_in.shape[0], dtype=np.int64) * j)
            feat_in_list.append(f_in)
            trun_in_list.append(trun_in)
        #         cam_in_feat_arr = np.concatenate(cam_in_feat_list)
        #         track_in_feat_arr = np.concatenate(track_in_feat_arr)
        index_in_arr = np.concatenate(index_in_list)
        feat_in_arr = np.concatenate(feat_in_list)  # mxc
        trun_in_arr = np.concatenate(trun_in_list)  # m
        print('done for preparing feat_in_arr')

        print('start to compute distance matrix...')
        dist_mat = self.compute_distance_matrix(feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                                                cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id,
                                                cam_in_id,
                                                trun_out_arr, trun_in_arr)  # nxm

        print('start to find matched pairs...')
        matched_i, matched_j, matched_d = self.find_pairs(dist_mat, index_out_arr, index_in_arr,
                                                          track_out_arr, track_in_arr, out_time_out_arr, in_time_in_arr,
                                                          cam_out_id, cam_in_id)

        return matched_i, matched_j

    def compute_distance_matrix(self, feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                                cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id,
                                trun_out_arr, trun_in_arr):
        # dist_mat = np.matmul(feat_out_arr, feat_in_arr.T) # baseline for cosine
        # dist_mat = distance.cdist(feat_out_arr, feat_in_arr, 'euclidean') # baseline for L2

        # rerank
        q_q_sim = np.matmul(feat_out_arr, feat_out_arr.T)
        g_g_sim = np.matmul(feat_in_arr, feat_in_arr.T)
        q_g_sim = np.matmul(feat_out_arr, feat_in_arr.T)
        k1 = self.k1
        k2 = self.k2
        lambda_value = self.lambda_value
        dist_mat = re_ranking(q_g_sim, q_q_sim, g_g_sim, k1=k1, k2=k2, lambda_value=lambda_value)  # nxm
        # breakpoint()

        # mask with intervals
        tth_min, tth_max = two_track_valid_pass_time_for_mask_dict[(cam_out_id, cam_in_id)]
        out_time_out_box_arr = out_time_out_arr[index_out_arr]  # n
        in_time_in_box_arr = in_time_in_arr[index_in_arr]  # m
        n_out_box = out_time_out_box_arr.shape[0]
        n_in_box = in_time_in_box_arr.shape[0]
        out_time_out_box_mat = np.expand_dims(out_time_out_box_arr, 1).repeat(n_in_box, 1)  # nxm
        in_time_in_box_mat = np.expand_dims(in_time_in_box_arr, 0).repeat(n_out_box, 0)  # nxm

        alpha = self.alpha  # param need to be adapted
        long_time_t = self.long_time_t  # param need to be adapted
        short_time_t = self.short_time_t  # param need to be adapted
        travel_time_mat = in_time_in_box_mat - out_time_out_box_mat
        travel_time_mask = np.ones_like(travel_time_mat)
        too_short_pairs_indices = (travel_time_mat < tth_min)
        too_long_pairs_indices = (travel_time_mat > tth_max)
        if len(too_short_pairs_indices) > 0:
            travel_time_mask[too_short_pairs_indices] = np.exp(alpha * (tth_min - \
                                                                        travel_time_mat[
                                                                            too_short_pairs_indices]) / short_time_t)
        if len(too_long_pairs_indices) > 0:
            travel_time_mask[too_long_pairs_indices] = np.exp(alpha * (travel_time_mat[too_long_pairs_indices] \
                                                                       - tth_max) / long_time_t)

        # travel_time_mask[too_short_pairs_indices] = np.exp(alpha * -(travel_time_mat[too_short_pairs_indices] - tth_min) )
        # travel_time_mask[too_long_pairs_indices] = np.exp(alpha * -( tth_max - travel_time_mat[too_long_pairs_indices] ) )

        dist_mat *= travel_time_mask

        # mask with occlusion
        occ_rate = self.occ_rate
        occ_alpha = self.occ_alpha
        trun_out_arr = np.expand_dims(trun_out_arr, 1).repeat(n_in_box, 1)  # nxm
        trun_out_mask_arr = (trun_out_arr > occ_rate)
        trun_out_weight_arr = np.ones_like(trun_out_arr)
        trun_out_weight_arr[trun_out_mask_arr] = np.exp(occ_alpha * (1 + trun_out_arr[trun_out_mask_arr]))

        trun_in_arr = np.expand_dims(trun_in_arr, 0).repeat(n_out_box, 0)  # nxm
        trun_in_mask_arr = (trun_in_arr > occ_rate)
        trun_in_weight_arr = np.ones_like(trun_in_arr)
        trun_in_weight_arr[trun_in_mask_arr] = np.exp(occ_alpha * (1 + trun_in_arr[trun_in_mask_arr]))

        dist_mat *= trun_out_weight_arr
        dist_mat *= trun_in_weight_arr

        return dist_mat

    def find_pairs(self, dist_mat, index_out_arr, index_in_arr, track_out_arr, track_in_arr,
                   out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id):
        sorted_out_index_dist_mat = dist_mat.argsort(1)  # nxm
        sorted_in_index_dist_mat = dist_mat.argsort(0)  # nxm
        topk = self.topk  # param need to be adapted
        r_rate = self.r_rate  # param need to be adapted
        matched_box_dict = {}
        for i in range(dist_mat.shape[0]):  # iter out_port
            forward_candidate_track_index_arr = index_in_arr[sorted_out_index_dist_mat[i]][:topk]  # track_in_arr的index
            bin_count = np.bincount(forward_candidate_track_index_arr)
            forward_matched_track_id = np.argmax(bin_count)  # 匹配次数最多的tracklet的index
            forward_matched_track_id_count = bin_count[forward_matched_track_id]  # 该tracklet的出现次数

            if forward_matched_track_id_count < 2:
                continue
            indices = np.where(forward_candidate_track_index_arr == forward_matched_track_id)[0]
            forward_indices = sorted_out_index_dist_mat[i, indices]
            reverse_pair_track_count = 0
            for j in forward_indices:
                if i in sorted_in_index_dist_mat[:topk, j]:
                    reverse_pair_track_count += 1
            if reverse_pair_track_count / forward_matched_track_id_count > r_rate:
                track_index = index_out_arr[i]
                if track_index not in matched_box_dict:
                    matched_box_dict[track_index] = []
                matched_box_dict[track_index].append(forward_matched_track_id)

        # find pairs
        matched_i = []
        matched_j = []
        matched_d = []
        for track_out_index in sorted(matched_box_dict.keys()):
            bin_count = np.bincount(np.array(matched_box_dict[track_out_index]))
            track_in_index = np.argmax(bin_count)
            track_in_index_count = bin_count[track_in_index]
            if track_in_index_count > 1:
                matched_i.append(track_out_index)
                matched_j.append(track_in_index)
                matched_d.append(track_in_index_count)
        matched_i = np.array(matched_i, dtype=np.int16)
        matched_j = np.array(matched_j, dtype=np.int16)
        matched_d = np.array(matched_d, dtype=np.int16)

        # filter repeated pairs
        unique_matched_j = np.unique(matched_j)
        filtered_matched_i = []
        filtered_matched_j = []
        filtered_matched_d = []
        for mj in unique_matched_j:
            mj_arr = np.where(matched_j == mj)[0]
            if mj_arr.shape[0] > 1:
                md_arr = matched_d[mj_arr]
                md_index = md_arr.argmax()  # need to be optimized
                mj_index = mj_arr[md_index]
                filtered_matched_i.append(matched_i[mj_index])
                filtered_matched_j.append(matched_j[mj_index])
                filtered_matched_d.append(matched_d[mj_index])
            else:
                filtered_matched_i.append(matched_i[mj_arr[0]])
                filtered_matched_j.append(matched_j[mj_arr[0]])
                filtered_matched_d.append(matched_d[mj_arr[0]])
        filtered_matched_i = np.array(filtered_matched_i, dtype=np.int16)
        filtered_matched_j = np.array(filtered_matched_j, dtype=np.int16)
        filtered_matched_d = np.array(filtered_matched_d, dtype=np.int16)
        return filtered_matched_i, filtered_matched_j, filtered_matched_d

    def drop_invalid_matched_pairs(self, matched_i, matched_j, cam_out_id, cam_in_id, out_time_out_arr, in_time_in_arr):
        """
        Args:
            matched_i:
            matched_j:
        """
        tth_min, tth_max = two_track_valid_pass_time_dict[(cam_out_id, cam_in_id)]
        keep_ids = []
        for idx, (i, j) in enumerate(zip(matched_i, matched_j)):
            travel_time = in_time_in_arr[j] - out_time_out_arr[i]
            if travel_time < tth_min or travel_time > tth_max:
                continue
            keep_ids.append(idx)
        matched_i = matched_i[keep_ids]
        matched_j = matched_j[keep_ids]
        return matched_i, matched_j

    def matching(self, cam_in_id, cam_out_id, interval_out=[0, 2001], interval_in=[0, 2001], \
                 direction=True, mode='linear', st_dim=0, en_dim=2048, is_params=True):

        map_out_arr, map_in_arr = self.select_map_arr2(cam_out_id, cam_in_id)

        cam_out_arr = self.cam_arr[map_out_arr]
        track_out_arr = self.track_arr[map_out_arr]
        # in_dir_out_arr = self.in_dir_arr[map_out_arr]
        # out_dir_out_arr = self.out_dir_arr[map_out_arr]
        in_time_out_arr = self.in_time_arr[map_out_arr]
        out_time_out_arr = self.out_time_arr[map_out_arr]  # - time_stamps[cam_out_id]

        cam_in_arr = self.cam_arr[map_in_arr]
        track_in_arr = self.track_arr[map_in_arr]
        # in_dir_in_arr = self.in_dir_arr[map_in_arr]
        # out_dir_in_arr = self.out_dir_arr[map_in_arr]
        in_time_in_arr = self.in_time_arr[map_in_arr]  # - time_stamps[cam_in_id]
        out_time_in_arr = self.out_time_arr[map_in_arr]
        print('cam: {}; tracks: {} \t track_ids: {}'.format(cam_out_id, len(track_out_arr), np.sort(track_out_arr)))
        print('cam: {}; tracks: {} \t track_ids: {}'.format(cam_in_id, len(track_in_arr), np.sort(track_in_arr)))

        print('cam: {}; tracks: {} \t track_time: {}'.format(cam_out_id, len(out_time_out_arr),
                                                             np.sort(out_time_out_arr)))
        print('cam: {}; tracks: {} \t track_time: {}'.format(cam_in_id, len(in_time_in_arr), np.sort(in_time_in_arr)))

        #### Search results circularly
        all_matched_i = []
        all_matched_j = []
        print('* Start matching...')
        topk = self.topk
        r_rate = self.r_rate
        num_search_times = self.num_search_times
        for i in range(num_search_times):
            print('** Iter {}...'.format(i))
            sub_track_out_arr = np.setdiff1d(track_out_arr, track_out_arr[all_matched_i])  # sorted. need to be readjust
            sub_track_in_arr = np.setdiff1d(track_in_arr, track_in_arr[all_matched_j])  # sorted. need to be readjust

            num_candidates = 1
            if sub_track_out_arr.shape[0] < num_candidates or sub_track_in_arr.shape[0] < num_candidates:
                break

            map_sub_out_arr = np.isin(track_out_arr, sub_track_out_arr, True)  # accelerate
            map_sub_in_arr = np.isin(track_in_arr, sub_track_in_arr, True)  # accelerate

            sub_track_out_arr = track_out_arr[map_sub_out_arr]  # original order
            sub_track_in_arr = track_in_arr[map_sub_in_arr]  # original order

            sub_cam_out_arr = cam_out_arr[map_sub_out_arr]
            sub_cam_in_arr = cam_in_arr[map_sub_in_arr]

            sub_out_time_out_arr = out_time_out_arr[map_sub_out_arr]
            sub_in_time_in_arr = in_time_in_arr[map_sub_in_arr]
            # breakpoint()
            r = min(sub_track_out_arr.shape[0] / float(track_out_arr.shape[0]),
                    sub_track_in_arr.shape[0] / float(track_in_arr.shape[0]))
            self.topk = int(topk * r)
            if self.topk < 3:
                self.topk = 3

            sub_matched_i, sub_matched_j = self.do_matching(sub_cam_out_arr, sub_cam_in_arr, sub_track_out_arr,
                                                            sub_track_in_arr, sub_out_time_out_arr, sub_in_time_in_arr,
                                                            cam_out_id, cam_in_id, st_dim=st_dim, en_dim=en_dim)
            print('matching results:')
            print(sub_matched_i, sub_matched_j)

            sub_matched_i, sub_matched_j = self.drop_invalid_matched_pairs(sub_matched_i, sub_matched_j,
                                                                           cam_out_id, cam_in_id, sub_out_time_out_arr,
                                                                           sub_in_time_in_arr)

            for smi, smj in zip(sub_matched_i, sub_matched_j):
                # assert mi, mj only match one item in track_arr
                mi = np.where(track_out_arr == sub_track_out_arr[smi])[0].item()
                mj = np.where(track_in_arr == sub_track_in_arr[smj])[0].item()

                assert (mi not in all_matched_i)
                assert (mj not in all_matched_j)
                all_matched_i.append(mi)
                all_matched_j.append(mj)
        try:
            matched_i = np.array(all_matched_i, dtype=int)
            matched_j = np.array(all_matched_j, dtype=int)
        except Exception as ex:
            print(all_matched_i, all_matched_j)
            raise ex
        try:
            matched_track_out_arr = track_out_arr[matched_i]
        except Exception as ex:
            print(matched_i, track_out_arr)
            raise ex

        sorted_ids = np.argsort(matched_track_out_arr)
        matched_i = matched_i[sorted_ids]  # for print
        matched_j = matched_j[sorted_ids]  # for print

        print('number of matched pairs: {}'.format(len(matched_i)))
        global_max_id = self.global_id_arr.max() + 1
        for i, j in zip(matched_i, matched_j):
            track_out_id = track_out_arr[i]
            track_in_id = track_in_arr[j]
            idx_i = (self.cam_arr == cam_out_id) & (self.track_arr == track_out_id)
            idx_j = (self.cam_arr == cam_in_id) & (self.track_arr == track_in_id)

            try:
                assert (self.global_id_arr[idx_j].item() == -1)
            except:
                pdb.set_trace()
            if self.global_id_arr[idx_i].item() != -1:
                self.global_id_arr[idx_j] = self.global_id_arr[idx_i]
            else:
                self.global_id_arr[idx_i] = global_max_id
                self.global_id_arr[idx_j] = global_max_id
                global_max_id += 1
            all_g_ids = np.where(self.global_id_arr == self.global_id_arr[idx_i].item())[0]
            all_matched_cams = self.cam_arr[all_g_ids]
            all_matched_tracks = self.track_arr[all_g_ids]
            print('{:3d}: ({:3d}, {:3d}) \t interval: {:4d} \t all_matched_cams: {:18s} \t '
                  'all_matched_tracks: {}'.format(self.global_id_arr[idx_i].item(),
                                                  track_out_id, track_in_id,
                                                  self.in_time_arr[idx_j].item() - self.out_time_arr[idx_i].item(),
                                                  ', '.join(map(str, all_matched_cams)),
                                                  ', '.join(map(str, all_matched_tracks))))

    def forward_matching(self, mode='linear', st_dim=0, en_dim=2048, is_test=False, cam_out_id=10, cam_in_id=11):
        if is_test:
            print('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            self.matching(cam_in_id, cam_out_id, direction=True, mode=mode, \
                          st_dim=st_dim, en_dim=en_dim, is_params=False)
            return

        # positve matching
        for cam_out_id, cam_in_id in path1:
            print('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            # key = (cam_out_id, cam_in_id)
            # print('params: {}'.format(args_params_dict[key]))
            # self.topk = args_params_dict[key]['topk']
            # self.r_rate = args_params_dict[key]['r_rate']
            # self.k1 = args_params_dict[key]['k1']
            # self.k2 = args_params_dict[key]['k2']
            # self.lambda_value = args_params_dict[key]['lambda_value']
            # self.alpha = args_params_dict[key]['alpha']
            # self.long_time_t = args_params_dict[key]['long_time_t']
            # self.short_time_t = args_params_dict[key]['short_time_t']
            # self.num_search_times = args_params_dict[key]['num_search_times']

            self.matching(cam_in_id, cam_out_id, direction=True, mode=mode, \
                          st_dim=st_dim, en_dim=en_dim, is_params=False)

        # negative matching
        for cam_out_id, cam_in_id in path2:
            print('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            # key = (cam_out_id, cam_in_id)
            # print('params: {}'.format(args_params_dict[key]))
            # self.topk = args_params_dict[key]['topk']
            # self.r_rate = args_params_dict[key]['r_rate']
            # self.k1 = args_params_dict[key]['k1']
            # self.k2 = args_params_dict[key]['k2']
            # self.lambda_value = args_params_dict[key]['lambda_value']
            # self.alpha = args_params_dict[key]['alpha']
            # self.long_time_t = args_params_dict[key]['long_time_t']
            # self.short_time_t = args_params_dict[key]['short_time_t']

            self.matching(cam_in_id, cam_out_id, direction=False, mode=mode, \
                          st_dim=st_dim, en_dim=en_dim, is_params=False)

    def write_output(self, src_path, dst_path):
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

        print('* writing output...')
        dst_obj = open(dst_path, 'w')
        with open(src_path, 'r') as fid:
            for line in fid.readlines():
                s = [int(i) for i in line.rstrip().split()]

                idx = ((self.cam_arr == s[0]) & (self.track_arr == s[1]))
                g_id = self.global_id_arr[idx].item()
                if g_id != -1:
                    s[1] = g_id
                    dst_obj.write(
                        '{}\n'.format(' '.join(map(str, s))))  # [camera_id, track_id, frame_id, x, y, w, h, -1, -1]
        dst_obj.close()

    def fill_global_id_arr(self):
        max_id = self.global_id_arr.max() + 1
        for i in range(len(self.global_id_arr)):
            if self.global_id_arr[i] == -1:
                self.global_id_arr[i] = max_id + 1
                max_id += 1



class Tracklet(object):
    def __init__(self, cam_id, x, y, fr_id, xywh):
        self.cam_id = cam_id
        self.frames = [fr_id]
        self.bboxes = [xywh]
        self.st_id = -1  # the "in port" id when the track appears in the image at the start
        self.en_id = -1  # the "out port" id when the track disappears in the image at the end
        self.select_st_id(x, y)

    def select_st_id(self, x, y):
        self.st_id = mask_matrix[self.cam_id][x][y]

    def select_en_id(self, x, y):
        self.en_id = mask_matrix[self.cam_id][x][y]

    def add_element(self, x, y, fr_id, xywh):
        self.bboxes.append(xywh)
        self.frames.append(fr_id)
        self.select_en_id(x, y)


def calc_occlusion_score(bboxes, track_ids):
    """
    Calculate occlusion score
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2] + x1
    y2 = bboxes[:, 3] + y1
    scores = y2
    sorted_ids = scores.argsort()[::-1]
    areas = (x2 - x1) * (y2 - y1)

    res_dict = {}

    for j in range(len(sorted_ids)):
        i = sorted_ids[j]
        xx1 = np.maximum(x1[i], x1[sorted_ids[j + 1:]])
        yy1 = np.maximum(y1[i], y1[sorted_ids[j + 1:]])
        xx2 = np.minimum(x2[i], x2[sorted_ids[j + 1:]])
        yy2 = np.minimum(y2[i], y2[sorted_ids[j + 1:]])

        inter_areas = np.maximum(0., xx2 - xx1 + 1) * np.maximum(0., yy2 - yy1 + 1)
        ious = inter_areas / areas[sorted_ids[j + 1:]]
        max_iou = np.max(ious) if ious.size != 0 else 0

        # key = 'c0{}_{}_{}'.format(cam_id, fr_id, track_ids[i])
        res_dict[track_ids[i]] = max_iou

    return res_dict


mask_matrix = {}
_path = input_path + '/videos'
for folder in os.listdir(_path):
    mask = cv2.imread(_path + '/' + folder + '/' + 'mask_zone.jpg', cv2.IMREAD_GRAYSCALE)
    s = folder.split('.')[0][-2:]
    s = int(s)
    mask = np.where(mask > 5, 0, mask)
    mask_matrix[s] = mask

with open(output_pkl_path + '/results_scmt.pkl', 'rb') as f:
    cam_dict_results = pickle.load(f)


def run():
    cam_dict_tracklet = {}
    for cam_id, tracks in cam_dict_results.items():
        cam_dict_tracklet.setdefault(cam_id, {})
        for track in tracks:
            t, l, w, h = track[2:6]
            yc = int(t + w // 2)
            xc = int(l + h // 2)
            if track[1] not in cam_dict_tracklet[cam_id].keys():
                cam_dict_tracklet[cam_id].setdefault(int(track[1]), Tracklet(cam_id, xc, yc, track[0], (t, l, w, h)))
            else:
                cam_dict_tracklet[cam_id][int(track[1])].add_element(xc, yc, track[0], (t, l, w, h))

    trun_dict = {}
    feat_dict = {}
    for cam_id, tracks in cam_dict_results.items():
        feat_dict[cam_id] = {}
        trun_dict[cam_id] = {}

        bboxes = []
        track_ids = []
        last_frame = tracks[0][0]
        res = {}

        for track in tracks:
            if track[1] not in feat_dict[cam_id].keys():
                feat_dict[cam_id][int(track[1])] = []
            if track[1] not in trun_dict[cam_id].keys():
                trun_dict[cam_id][int(track[1])] = []

            if track[0] != last_frame:
                # print(np.array(bboxes).shape, last_frame, key, track_ids, last_frame)
                res = calc_occlusion_score(np.array(bboxes), track_ids)
                for tid in res.keys():
                    trun_dict[cam_id][tid].append(res[tid])
                bboxes = []
                track_ids = []
                last_frame = track[0]

            bboxes.append(track[2:6])
            track_ids.append(track[1])

            feat_dict[cam_id][int(track[1])].append(track[7])

        for tid in res.keys():
            trun_dict[cam_id][tid].append(res[tid])

    track_cam_id_arr, track_id_arr, track_st_zone, track_en_zone, track_st_frame, track_en_frame = \
        [], [], [], [], [], []

    for cid in cam_dict_tracklet.keys():
        for tid in cam_dict_tracklet[cid].keys():
            track_cam_id_arr.append(int(cid))
            track_id_arr.append(int(tid))
            track_st_zone.append(int(cam_dict_tracklet[cid][tid].st_id))
            track_en_zone.append(int(cam_dict_tracklet[cid][tid].en_id))
            track_st_frame.append(int(cam_dict_tracklet[cid][tid].frames[0]))
            track_en_frame.append(int(cam_dict_tracklet[cid][tid].frames[-1]))

    # cutting tracks that are too short
    length = len(track_cam_id_arr)
    i = 0
    while i < length:
        if track_en_frame[i] - track_st_frame[i] < 4:  # or track_en_frame[i] == cam_frames[track_cam_id_arr[i]]:
            track_cam_id_arr.pop(i)
            track_id_arr.pop(i)
            track_st_zone.pop(i)
            track_en_zone.pop(i)
            track_st_frame.pop(i)
            track_en_frame.pop(i)

            i -= 1
            length -= 1
        i += 1

    track_cam_id_arr = np.array(track_cam_id_arr)
    track_id_arr = np.array(track_id_arr)
    track_st_zone = np.array(track_st_zone)
    track_en_zone = np.array(track_en_zone)
    track_st_frame = np.array(track_st_frame)
    track_en_frame = np.array(track_en_frame)

    with open('all_cameras_scmt.txt', 'w') as f:
        for i in range(len(track_cam_id_arr)):
            cid = track_cam_id_arr[i]
            tid = track_id_arr[i]
            for j in range(len(cam_dict_tracklet[cid][tid].frames)):
                frame = cam_dict_tracklet[cid][tid].frames[j]
                xywh = cam_dict_tracklet[cid][tid].bboxes[j]
                f.write(f"{cid} {tid} {frame} {int(xywh[0])} {int(xywh[1])} {int(xywh[2])} {int(xywh[3])} -1 -1\n")

    matcher = MultiCameraMatching(track_cam_id_arr, track_id_arr, track_st_zone, track_en_zone,
                                  track_st_frame, track_en_frame, feat_dict,
                                  topk=12, r_rate=0.5,
                                  k1=12, k2=7, lambda_value=0.6,
                                  alpha=1.1, long_time_t=500, short_time_t=500,
                                  num_search_times=2,
                                  trun_dict=trun_dict, occ_rate=1.0, occ_alpha=0.)

    matcher.forward_matching(mode='linear', st_dim=0, en_dim=2048, is_test=False, cam_out_id=13, cam_in_id=14)

    matcher.write_output('./all_cameras_scmt.txt', './output_ica.txt')
