#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from visualize.vicon_visualization import from_array
import utils.karate.data_info as data_info
from scipy import linalg
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_inception_embeddings
from utils.model_util import create_semantic_regressor, create_motion_classifier
from load.get_data import get_dataset_loader
from utils.parser_util import manipulation_quantitative_evaluation_with_inception_args
from utils.parser_util import model_parser, get_model_path_from_args
import torch
from evaluation.fid import calculate_fid
from sample.rejection_generation import generate_samples_for_data_loader, create_attribute_labels
import copy
import matplotlib.pyplot as plt
from sample.guided_manipulation import manipulate_single_sample

grade_number_to_name = {
    0: '9 kyu',
    1: '8 kyu',
    2: '7 kyu',
    3: '6 kyu',
    4: '5 kyu',
    5: '4 kyu',
    6: '3 kyu',
    7: '2 kyu',
    8: '1 kyu',
    9: '1 dan',
    10: '2 dan',
    11: '3 dan',
    12: '4 dan'
}

technique_class_to_name = {
    0: 'Reverse punch',   # reverse punch
    1: 'Front kick',   # front kick
    2: 'Low roundhouse kick',   # roundhouse kick at knee to hip height
    3: 'High roundhouse kick',   # roundhouse kick at shoulder to (top) head height
    4: 'Spinning back kick'   # spinning back kick
}


def load_modiffae(modiffae_model_path):
    modiffae_args = model_parser(model_type="modiffae", model_path=modiffae_model_path)

    modiffae_validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    modiffae_model, modiffae_diffusion = create_modiffae_and_diffusion(modiffae_args, modiffae_validation_data)

    print(f"Loading checkpoints from [{modiffae_model_path}]...")
    modiffae_state_dict = torch.load(modiffae_model_path, map_location='cpu')
    load_model(modiffae_model, modiffae_state_dict)

    modiffae_model.to(dist_util.dev())
    modiffae_model.eval()

    return modiffae_model, modiffae_diffusion, modiffae_args


def load_semantic_regressor_ckpt(semantic_regressor_model_path, semantic_encoder):
    semantic_regressor_args = model_parser(model_type="semantic_regressor", model_path=semantic_regressor_model_path)

    # Important to use train data here! Regressor calculates z parameters based on them.
    semantic_regressor_train_data = get_dataset_loader(
        name=semantic_regressor_args.dataset,
        batch_size=semantic_regressor_args.batch_size,
        num_frames=semantic_regressor_args.num_frames,
        test_participant=semantic_regressor_args.test_participant,
        pose_rep=semantic_regressor_args.pose_rep,
        split='train'
    )

    semantic_regressor_model = create_semantic_regressor(
        semantic_regressor_args,
        semantic_regressor_train_data,
        semantic_encoder
    )

    print(f"Loading checkpoints from [{semantic_regressor_model_path}]...")
    semantic_regressor_state_dict = torch.load(semantic_regressor_model_path, map_location='cpu')
    load_model(semantic_regressor_model, semantic_regressor_state_dict)

    semantic_regressor_model.to(dist_util.dev())
    semantic_regressor_model.eval()

    return semantic_regressor_model, semantic_regressor_args


def load_motion_classifier(motion_classifier_model_path):
    motion_classifier_args = model_parser(model_type="motion_classifier", model_path=motion_classifier_model_path)

    validation_data = get_dataset_loader(
        name=motion_classifier_args.dataset,
        batch_size=motion_classifier_args.batch_size,
        num_frames=motion_classifier_args.num_frames,
        test_participant=motion_classifier_args.test_participant,
        pose_rep=motion_classifier_args.pose_rep,
        split='validation'
    )

    motion_classifier_model = create_motion_classifier(motion_classifier_args, validation_data)

    print(f"Loading checkpoints from [{motion_classifier_model_path}]...")
    motion_classifier_state_dict = torch.load(motion_classifier_model_path, map_location='cpu')
    load_model(motion_classifier_model, motion_classifier_state_dict)

    motion_classifier_model.to(dist_util.dev())
    motion_classifier_model.eval()

    return motion_classifier_model, motion_classifier_args


def extract_data_for_labels(data, technique_cls, grade_name):
    data_loader_copy = copy.deepcopy(data)
    data_loader_copy.dataset.reduce_to_data_for_labels(technique_cls, grade_name)
    return data_loader_copy


def calc_fid_combinations(train_data, validation_data, original_grade_name, original_technique_cls,
                          motion_inception_model, test_embeddings):
    technique_combination_fid_list = []
    for tech_nr, tech_name in technique_class_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, tech_nr, original_grade_name)
        combination_comparison_data_embeddings = calculate_inception_embeddings(combination_comparison_data,
                                                                                motion_inception_model)

        technique_combination_fid = calculate_fid(test_embeddings,
                                                  combination_comparison_data_embeddings)
        technique_combination_fid_list.append(technique_combination_fid)

    # Calculate distances to all the grades of the current technique.
    grade_combination_fid_list = []
    for grade_nr, grade_name in grade_number_to_name.items():
        combination_comparison_data = extract_data_for_labels(validation_data, original_technique_cls, grade_name)
        combination_comparison_data_embeddings = calculate_inception_embeddings(combination_comparison_data,
                                                                                motion_inception_model)

        grade_combination_fid = calculate_fid(test_embeddings,
                                              combination_comparison_data_embeddings)
        grade_combination_fid_list.append(grade_combination_fid)

    return technique_combination_fid_list, grade_combination_fid_list


def main():
    args = manipulation_quantitative_evaluation_with_inception_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    modiffae_model_path = args.modiffae_model_path
    modiffae_model, modiffae_diffusion, modiffae_args = load_modiffae(modiffae_model_path)

    semantic_regressor_model_path = args.semantic_regressor_model_path
    semantic_regressor_model, _ = (
        load_semantic_regressor_ckpt(semantic_regressor_model_path, modiffae_model.semantic_encoder))

    motion_classifier_model_path = args.motion_classifier_model_path
    motion_inception_model, _ = \
        load_motion_classifier(motion_classifier_model_path)

    test_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='test'
    )

    validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    train_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='train'
    )

    if modiffae_args.test_participant == 'b0401':
        original_grade_nr = 1
        t_g = 9
        neg = False
    elif modiffae_args.test_participant == 'b0372':
        original_grade_nr = 9
        t_g = 1
        neg = True
    else:
        raise ValueError('Invalid test participant')

    # Specify which case to evaluate

    # Technique example (bad)
    '''original_technique_cls = 1  # To filter out the according techniques from the test participant
    #original_grade = None
    target_technique_idx = 0
    target_grade_nr = original_grade_nr #(1 / 12) * original_grade_nr
    manipulated_attribute_idx = 0
    is_negative = False
    batch_size = modiffae_args.batch_size'''

    # Technique example (good)
    original_technique_cls = 3  # To filter out the according techniques from the test participant
    # original_grade = None
    target_technique_idx = 1
    target_grade_nr = original_grade_nr  # (1 / 12) * original_grade_nr
    manipulated_attribute_idx = 1
    is_negative = False
    batch_size = modiffae_args.batch_size

    # Grade example (bad but justified)
    '''original_technique_cls = 1
    target_technique_idx = 1
    target_grade_nr = t_g
    manipulated_attribute_idx = 5
    is_negative = neg
    batch_size = modiffae_args.batch_size'''

    original_grade_name = grade_number_to_name[original_grade_nr]
    # Iterate over techniques and calculate the distance to the current technique
    combination_test_data_og = extract_data_for_labels(test_data, original_technique_cls, original_grade_name)
    combination_test_data_og_embeddings = calculate_inception_embeddings(combination_test_data_og,
                                                                         motion_inception_model)

    og_technique_combination_fid_list, og_grade_combination_fid_list = (
        calc_fid_combinations(train_data, validation_data, original_grade_name, original_technique_cls,
                              motion_inception_model, combination_test_data_og_embeddings))

    print(og_technique_combination_fid_list)
    print(og_grade_combination_fid_list)

    combination_test_manipulated_embeddings_after_motion_creation = []
    for b, (motion, cond) in enumerate(combination_test_data_og):
        motion = motion.to(dist_util.dev())
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}

        cond_labels = cond['y']['labels']

        for i in range(motion.shape[0]):
            single_model_kwargs = copy.deepcopy(cond)
            single_model_kwargs['y'].update({'mask': cond['y']['mask'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'lengths': cond['y']['lengths'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'original_motion': cond['y']['original_motion'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'distance': cond['y']['distance'][i].unsqueeze(dim=0)})
            single_model_kwargs['y'].update({'labels': cond['y']['labels'][i].unsqueeze(dim=0)})

            m = motion[i].unsqueeze(dim=0)

            target_grade = (1 / 12) * target_grade_nr

            og_xyz_motion, manipulated_xyz_motion, manipulated_sample = manipulate_single_sample(
                m, single_model_kwargs, modiffae_model, modiffae_diffusion, semantic_regressor_model,
                target_technique_idx=target_technique_idx, target_grade=target_grade,
                manipulated_attribute_idx=manipulated_attribute_idx,
                is_negative=is_negative, batch_size=batch_size, return_rot_sample=True
            )

            with torch.no_grad():
                manipulated_emb = modiffae_model.semantic_encoder(manipulated_sample)
            combination_test_manipulated_embeddings_after_motion_creation.append(manipulated_emb)

            '''from_array(arr=og_xyz_motion, mode='inspection')
            from_array(arr=manipulated_xyz_motion, mode='inspection')'''
    combination_test_manipulated_embeddings_after_motion_creation = (
        torch.cat(combination_test_manipulated_embeddings_after_motion_creation, dim=0))

    print(combination_test_manipulated_embeddings_after_motion_creation.shape)

    target_grade_name = grade_number_to_name[target_grade_nr]
    target_technique_cls = target_technique_idx

    manipulated_technique_combination_fid_list, manipulated_grade_combination_fid_list = (
        calc_fid_combinations(train_data, validation_data, target_grade_name, target_technique_cls,
                              motion_inception_model,
                              combination_test_manipulated_embeddings_after_motion_creation))

    print(manipulated_technique_combination_fid_list)
    print(manipulated_grade_combination_fid_list)


if __name__ == "__main__":
    main()
