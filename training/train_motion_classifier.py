import os
import json
from utils.fixseed import fixseed
from utils.parser_util import motion_classifier_train_args
from utils import dist_util
from training.motion_classifier_training_loop import MotionClassifierTrainLoop
from load.get_data import get_dataset_loader
from utils.model_util import create_motion_classifier
from training.train_platforms import TensorboardPlatform


def main():
    args = motion_classifier_train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir is unknown.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    train_platform = TensorboardPlatform(args.save_dir)
    train_platform.report_args(args, name='Args')

    dist_util.setup_dist(args.device)
    print(f"Device: {dist_util.dev()}")

    print("creating train data loader...")
    train_data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        test_participant=args.test_participant,
        pose_rep=args.pose_rep,
        split='train'
    )

    if args.dataset == "karate":
        print("creating validation data loader...")
        validation_data = get_dataset_loader(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            test_participant=args.test_participant,
            pose_rep=args.pose_rep,
            split='validation'
        )
    else:
        validation_data = None

    model = create_motion_classifier(args, train_data)

    model.to(dist_util.dev())
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    print("Training...")
    MotionClassifierTrainLoop(args, train_platform, model, train_data, validation_data).run_loop()

    train_platform.close()


if __name__ == "__main__":
    main()
