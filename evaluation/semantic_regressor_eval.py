from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, regression_evaluation_args
from utils.model_util import create_modiffae_and_diffusion, load_model, calculate_embeddings
from utils.model_util import create_semantic_regressor
from utils import dist_util
from load.get_data import get_dataset_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns
import colorcet as cc
import pandas as pd
import json
import torch.nn.functional as F


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


def calc_distance_score(val_prediction, val_target):
    technique_prediction = np.argmax(F.softmax(torch.tensor(val_prediction[:5]), dim=-1).cpu().detach().numpy())
    technique_target = np.argmax(val_target[:5])

    grade_prediction_float = torch.sigmoid(torch.tensor(val_prediction[5])).cpu().detach().numpy()
    grade_target_float = val_target[5]
    grade_mae = np.linalg.norm(grade_prediction_float - grade_target_float)
    grade_prediction = round(grade_prediction_float * 12)
    grade_target = round(grade_target_float * 12)

    if technique_prediction == technique_target:
        technique_acc = 1
    else:
        technique_acc = 0

    errors = [(technique_target, technique_acc), (grade_target, grade_mae)]

    predictions_and_targets = (
        (technique_prediction, technique_target),
        (grade_prediction, grade_target)
    )

    return errors, predictions_and_targets


def calc_checkpoint_metrics(semantic_regressor_model_path,
                            semantic_encoder,
                            modiffae_validation_data,
                            validation_labels):

    semantic_regressor_model, _ = (
        load_semantic_regressor_ckpt(semantic_regressor_model_path, semantic_encoder))

    semantic_generator_predictions = []
    for motion, cond in modiffae_validation_data:
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}
        with torch.no_grad():
            og_motion = cond['y']['original_motion']
        pred = semantic_regressor_model(og_motion)
        semantic_generator_predictions.append(pred)
    semantic_generator_predictions = torch.cat(semantic_generator_predictions, dim=0).cpu().detach().numpy()

    technique_predictions = []
    technique_targets = []
    grade_predictions = []
    grade_targets = []

    error_scores = []
    for i in range(semantic_generator_predictions.shape[0]):
        val_prediction = semantic_generator_predictions[i]
        val_target = validation_labels[i]
        error_score, predictions_and_targets = calc_distance_score(val_prediction, val_target)

        error_scores.append(error_score)

        technique_predictions.append(predictions_and_targets[0][0])
        technique_targets.append(predictions_and_targets[0][1])
        grade_predictions.append(predictions_and_targets[1][0])
        grade_targets.append(predictions_and_targets[1][1])

    predictions_and_targets_combined = (
        (technique_predictions, technique_targets),
        (grade_predictions, grade_targets)
    )

    technique_accuracies = []
    for cls in range(5):
        tech_scores = [ac for (c, ac), _ in error_scores if c == cls]
        tech_scores_avg = np.mean(tech_scores)
        technique_accuracies.append(tech_scores_avg)

    grade_maes = []
    for gr in range(13):
        grade_scores = [mae for _, (g, mae) in error_scores if g == gr]
        grade_scores_avg = np.mean(grade_scores)
        grade_maes.append(grade_scores_avg)

    return technique_accuracies, grade_maes, predictions_and_targets_combined


def run_validation(validation_data, model):
    technique_accuracies_list = []
    grade_maes_list = []
    predictions_and_targets_combined = (([], []), ([], []))

    for motion, cond in validation_data:
        cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                     cond['y'].items()}
        technique_accuracies_batch, grade_maes_batch, predictions_and_targets_batch = (
            forward(cond, model)
        )
        technique_accuracies_list.extend(technique_accuracies_batch)
        grade_maes_list.extend(grade_maes_batch)

        predictions_and_targets_combined[0][0].extend(predictions_and_targets_batch[0][0])
        predictions_and_targets_combined[0][1].extend(predictions_and_targets_batch[0][1])
        predictions_and_targets_combined[1][0].extend(predictions_and_targets_batch[1][0])
        predictions_and_targets_combined[1][1].extend(predictions_and_targets_batch[1][1])

    technique_accuracies = []
    for cls in range(5):
        tech_scores = [ac for (c, ac) in technique_accuracies_list if c == cls]
        tech_scores_avg = np.mean(tech_scores)
        technique_accuracies.append(tech_scores_avg)

    grade_maes = []
    for gr in range(13):
        grade_scores = [mae for (g, mae) in grade_maes_list if g == gr]
        grade_scores_avg = np.mean(grade_scores)
        grade_maes.append(grade_scores_avg)

    return technique_accuracies, grade_maes, predictions_and_targets_combined


def calc_scores(technique_prediction, technique_target, grade_prediction_float, grade_target_float):
    if technique_prediction == technique_target:
        tech_acc = 1
    else:
        tech_acc = 0

    grade_mae = np.linalg.norm(grade_prediction_float - grade_target_float)

    grade_pred = round(grade_prediction_float * 12)
    grade_targ = round(grade_target_float * 12)

    pred_and_targ = (
        (technique_prediction, technique_target),
        (grade_pred, grade_targ)
    )

    return (technique_target, tech_acc), (grade_targ, grade_mae), pred_and_targ


def forward(cond, model):

    og_motion = cond['y']['original_motion']
    target = cond['y']['labels'].squeeze()

    with torch.no_grad():
        output = model(og_motion)

    action_output = output[:, :5]
    action_output = F.softmax(action_output, dim=-1)

    skill_level_output = output[:, 5]
    skill_level_output = torch.sigmoid(skill_level_output)

    action_target = target[:, :5]
    skill_level_target = target[:, 5]

    action_classifications = torch.argmax(action_output, dim=-1)
    action_labels_idxs = torch.argmax(action_target, dim=-1)

    technique_predictions = list(action_classifications.cpu().detach().numpy())
    technique_targets = list(action_labels_idxs.cpu().detach().numpy())

    grade_predictions_float = list(skill_level_output.cpu().detach().numpy())
    grade_targets_float = list(skill_level_target.cpu().detach().numpy())

    technique_accuracies_batch = []
    grade_maes_batch = []

    grade_predictions = []
    grade_targets = []

    for i in range(len(grade_predictions_float)):
        tech_acc, grade_mae, pred_and_targ = calc_scores(
            technique_predictions[i],
            technique_targets[i],
            grade_predictions_float[i],
            grade_targets_float[i]
        )
        technique_accuracies_batch.append(tech_acc)
        grade_maes_batch.append(grade_mae)
        grade_predictions.append(pred_and_targ[1][0])
        grade_targets.append(pred_and_targ[1][1])

    predictions_and_targets_combined = (
        (technique_predictions, technique_targets),
        (grade_predictions, grade_targets)
    )

    return technique_accuracies_batch, grade_maes_batch, predictions_and_targets_combined


def main():
    args = regression_evaluation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    modiffae_model_path = args.modiffae_model_path
    modiffae_model, modiffae_diffusion, modiffae_args = load_modiffae(modiffae_model_path)
    semantic_encoder = modiffae_model.semantic_encoder

    modiffae_validation_data = get_dataset_loader(
        name=modiffae_args.dataset,
        batch_size=modiffae_args.batch_size,
        num_frames=modiffae_args.num_frames,
        test_participant=modiffae_args.test_participant,
        pose_rep=modiffae_args.pose_rep,
        split='validation'
    )

    _, model_name = os.path.split(modiffae_model_path)
    model_name = model_name.split('.')[0]
    base_dir, _ = os.path.split(os.path.dirname(modiffae_model_path))
    test_participant = modiffae_args.test_participant

    semantic_regressor_dir = (
        os.path.join(base_dir, f'semantic_regressor_based_on_modiffae_{test_participant}_{model_name}'))

    checkpoints = [p for p in sorted(os.listdir(semantic_regressor_dir)) if p.startswith('model') and p.endswith('.pt')]

    technique_accuracies_all = []
    grade_maes_all = []
    predictions_and_targets_all = []
    for ch in checkpoints:
        semantic_regressor_model_path = os.path.join(semantic_regressor_dir, ch)

        semantic_regressor_model, _ = (
            load_semantic_regressor_ckpt(semantic_regressor_model_path, semantic_encoder))

        technique_accuracies, grade_maes, predictions_and_targets_combined = run_validation(
            modiffae_validation_data,
            semantic_regressor_model
        )

        technique_accuracies_all.append(technique_accuracies)
        print(technique_accuracies)

        grade_maes_all.append(grade_maes)
        predictions_and_targets_all.append(predictions_and_targets_combined)

    technique_accuracies_all = np.array(technique_accuracies_all)
    grade_maes_all = np.array(grade_maes_all)

    checkpoints = [str(int(int(ch.strip("model").strip(".pt")) / 1000)) + "K" for ch in checkpoints]

    technique_idx_to_name = {
        0: "ACC: Reverse punch",
        1: "ACC: Front kick",
        2: "ACC: Low roundhouse kick",
        3: "ACC: High roundhouse kick",
        4: "ACC: Spinning back kick"
    }

    technique_idx_to_name_short = {
        0: "RP",
        1: "FK",
        2: "LRK",
        3: "HRK",
        4: "SBK"
    }

    grade_idx_to_name = {
        0: 'MAE: 9 kyu',
        1: 'MAE: 8 kyu',
        2: 'MAE: 7 kyu',
        3: 'MAE: 6 kyu',
        4: 'MAE: 5 kyu',
        5: 'MAE: 4 kyu',
        6: 'MAE: 3 kyu',
        7: 'MAE: 2 kyu',
        8: 'MAE: 1 kyu',
        9: 'MAE: 1 dan',
        10: 'MAE: 2 dan',
        11: 'MAE: 3 dan',
        12: 'MAE: 4 dan'
    }

    grade_idx_to_name_short = {
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

    f = plt.figure()
    f.set_figwidth(18)
    f.set_figheight(8)

    plt.rc('font', size=20)
    plt.rc('legend', fontsize=16)

    x = [int(ch[:-1]) * 1000 for ch in checkpoints]

    for idx in range(technique_accuracies_all.shape[1]):
        y = technique_accuracies_all[:, idx]
        plt.plot(x, y, label=f"{technique_idx_to_name[idx]}")

    technique_unweighted_average_recalls = []
    for idx in range(technique_accuracies_all.shape[0]):
        technique_unweighted_average_recalls.append(np.mean(technique_accuracies_all[idx, :]))
    best_technique_avg_idx = np.argmax(technique_unweighted_average_recalls)
    best_technique_avg_x = best_technique_avg_idx * 5000

    plt.plot(x, technique_unweighted_average_recalls, label=f"UAR", color='black')

    plt.vlines(x=[best_technique_avg_x], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best UAR')


    plt.legend(loc='lower center', bbox_to_anchor=(.73, .02))
    plt.xlabel('Training steps')

    desired_x_ticks = [l * 50000 for l in list(range(11))]
    desired_labels = [str(int(ch / 1000)) + 'K' for ch in desired_x_ticks]

    plt.xticks(ticks=desired_x_ticks, labels=desired_labels)

    eval_dir = os.path.join(args.save_dir, "evaluation")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    fig_save_path = os.path.join(eval_dir, f"regression_technique_uar_{test_participant}")
    plt.savefig(fig_save_path)

    plt.clf()

    plt.rc('legend', fontsize=12)

    with sns.color_palette(cc.glasbey, n_colors=14):
        for idx in reversed(list(range(grade_maes_all.shape[1]))):
            y = grade_maes_all[:, idx]
            plt.plot(x, y, label=f"{grade_idx_to_name[idx]}")

    grade_averages = []
    for idx in range(grade_maes_all.shape[0]):
        grade_averages.append(np.mean(grade_maes_all[idx, :]))
    best_grade_avg_idx = np.argmin(grade_averages)
    best_grade_avg_x = best_grade_avg_idx * 5000

    plt.plot(x, grade_averages, label=f"UMAE", color='black')

    plt.vlines(x=[best_grade_avg_x], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best UMAE')

    plt.legend(loc='lower center', bbox_to_anchor=(.87, .38))

    desired_x_ticks = [l * 50000 for l in list(range(11))]
    desired_labels = [str(int(ch / 1000)) + 'K' for ch in desired_x_ticks]

    plt.xticks(ticks=desired_x_ticks, labels=desired_labels)

    plt.xlabel('Training steps')

    fig_save_path = os.path.join(eval_dir, f"regression_grade_umae_{test_participant}")
    plt.savefig(fig_save_path)

    plt.clf()

    f = plt.figure()
    f.set_figwidth(18)
    f.set_figheight(8)

    plt.rc('legend', fontsize=16)

    grade_averages_acc = [1 - avg for avg in grade_averages]
    combined_metric = (np.array(grade_averages_acc) + np.array(technique_unweighted_average_recalls)) / 2
    best_combined_avg_idx = np.argmax(combined_metric)
    best_combined_avg_x = best_combined_avg_idx * 5000

    plt.plot(x, technique_unweighted_average_recalls, label=f"UAR")
    plt.plot(x, grade_averages, label=f"UMAE")
    plt.plot(x, combined_metric, label=f"Combined score", color='black')

    plt.vlines(x=[best_combined_avg_x], ymin=0, ymax=1, colors='black', ls='--', lw=2,
               label='Best combined score')

    plt.legend(loc='lower center', bbox_to_anchor=(.81, .4))

    plt.xlabel('Training steps')

    desired_x_ticks = [l * 50000 for l in list(range(11))]
    desired_labels = [str(int(ch / 1000)) + 'K' for ch in desired_x_ticks]

    plt.xticks(ticks=desired_x_ticks, labels=desired_labels)

    fig_save_path = os.path.join(eval_dir, f"regression_combined_{test_participant}")
    plt.savefig(fig_save_path)

    plt.rc('font', size=12)
    #plt.rc('font', size=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    sns.set(font_scale=1.0)

    chosen_model_predictions_and_targets = predictions_and_targets_all[best_combined_avg_idx][0]

    technique_confusion_matrix_values = confusion_matrix(
        chosen_model_predictions_and_targets[1], chosen_model_predictions_and_targets[0]
    )

    df_cm = pd.DataFrame(technique_confusion_matrix_values,
                         index=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()],
                         columns=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()])

    #plt.figure(figsize=(10, 7))
    plt.figure(figsize=(10, 6))
    s = sns.heatmap(df_cm, annot=True, cmap='Blues')
    s.set_xlabel('Predicted technique')
    s.set_ylabel('True technique')

    fig_save_path = os.path.join(eval_dir, f"regression_best_combined_technique_confusion_matrix_{test_participant}")
    plt.savefig(fig_save_path)

    chosen_model_predictions_and_targets = predictions_and_targets_all[best_combined_avg_idx][1]

    grade_confusion_matrix_values = confusion_matrix(
        chosen_model_predictions_and_targets[1], chosen_model_predictions_and_targets[0]
    )

    df_cm = pd.DataFrame(grade_confusion_matrix_values,
                         index=[grade_idx_to_name_short[i] for i in grade_idx_to_name_short.keys()],
                         columns=[grade_idx_to_name_short[i] for i in grade_idx_to_name_short.keys()])

    #plt.figure(figsize=(10, 7))
    plt.figure(figsize=(10, 6))
    s = sns.heatmap(df_cm, annot=True, cmap='Blues')
    s.set_xlabel('Predicted grade')
    s.set_ylabel('True grade')
    fig_save_path = os.path.join(eval_dir, f"regression_best_combined_grade_confusion_matrix_{test_participant}")
    plt.savefig(fig_save_path)

    best_results = {
        "best technique checkpoint": str(checkpoints[best_technique_avg_idx]),
        "UAR of best technique checkpoint": str(technique_unweighted_average_recalls[best_technique_avg_idx]),
        "best grade checkpoint": str(checkpoints[best_grade_avg_idx]),
        "UMAE of best grade checkpoint": str(grade_averages[best_grade_avg_idx]),
        "best combined checkpoint": str(checkpoints[best_combined_avg_idx]),
        "UAR of best combined checkpoint": str(technique_unweighted_average_recalls[best_combined_avg_idx]),
        "UMAE of best combined checkpoint": str(grade_averages[best_combined_avg_idx]),
        "overall score of best combined checkpoint": str(combined_metric[best_combined_avg_idx])
    }

    best_results_save_path = os.path.join(eval_dir, f"regression_best_results_overview_{test_participant}.json")
    with open(best_results_save_path, 'w') as outfile:
        json.dump(best_results, outfile)


if __name__ == "__main__":
    main()
