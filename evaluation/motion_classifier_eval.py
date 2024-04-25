from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import model_parser, motion_classifier_evaluation_args
from utils.model_util import load_model, create_motion_classifier
from utils import dist_util
from load.get_data import get_dataset_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import colorcet as cc
import pandas as pd
import json
import torch.nn.functional as F


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
    args = motion_classifier_evaluation_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    motion_classifier_dir = args.save_dir

    checkpoints = [p for p in sorted(os.listdir(motion_classifier_dir)) if p.startswith('model') and p.endswith('.pt')]

    tmp_model_name = os.path.join(motion_classifier_dir, checkpoints[0])
    _, motion_classifier_args = load_motion_classifier(tmp_model_name)

    validation_data = get_dataset_loader(
        name=motion_classifier_args.dataset,
        batch_size=motion_classifier_args.batch_size,
        num_frames=motion_classifier_args.num_frames,
        test_participant=motion_classifier_args.test_participant,
        pose_rep=motion_classifier_args.pose_rep,
        split='validation'
    )

    test_participant = motion_classifier_args.test_participant

    technique_accuracies_all = []
    grade_maes_all = []
    predictions_and_targets_all = []
    for ch in checkpoints:
        motion_classifier_model_path = os.path.join(motion_classifier_dir, ch)

        motion_classifier_ckpt_model, _ = load_motion_classifier(motion_classifier_model_path)

        technique_accuracies, grade_maes, predictions_and_targets_combined = run_validation(
            validation_data,
            motion_classifier_ckpt_model
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

    desired_x_ticks = [l * 2000 for l in list(range(11))]
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

    plt.rc('font', size=10)

    sns.set(font_scale=1.0)

    chosen_model_predictions_and_targets = predictions_and_targets_all[best_combined_avg_idx][0]

    technique_confusion_matrix_values = confusion_matrix(
        chosen_model_predictions_and_targets[1], chosen_model_predictions_and_targets[0]
    )

    df_cm = pd.DataFrame(technique_confusion_matrix_values,
                         index=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()],
                         columns=[technique_idx_to_name_short[i] for i in technique_idx_to_name_short.keys()])

    plt.figure(figsize=(10, 7))
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

    plt.figure(figsize=(10, 7))
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
