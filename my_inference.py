import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv


def main():
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-m', '--metric_names', type=str, nargs='+', default=['PSNR'], help='IQA metric names, case sensitive.')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    args = parser.parse_args()

    # set up IQA models for each metric
    iqa_models = {metric_name.lower(): create_metric(metric_name.lower(), metric_mode=args.metric_mode) for metric_name in args.metric_names}

    if os.path.isfile(args.input):
        input_paths = [args.input]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

    if args.save_file:
        # mkdir
        sub_path = os.path.dirname(args.save_file)
        save_dir = os.path.join('/home/yuan0520/Warehouse/My-IQA/results', sub_path)
        os.makedirs(save_dir, exist_ok=True)

        sf = open(os.path.join('/home/yuan0520/Warehouse/My-IQA/results', sub_path, args.save_file.split('/')[-1]), 'w', newline='')
        sfwriter = csv.writer(sf)

    avg_scores = {metric_name: 0 for metric_name in args.metric_names}
    test_img_num = len(input_paths)

    pbar = tqdm(total=test_img_num, unit='image')
    for idx, img_path in enumerate(input_paths):
        img_name = os.path.basename(img_path)
        if args.metric_mode == 'FR':
            ref_img_path = ref_paths[idx]
        else:
            ref_img_path = None

        row = [img_name]
        for metric_name, iqa_model in iqa_models.items():
            if metric_name != 'fid':
                score = iqa_model(img_path, ref_img_path).cpu().item()
                avg_scores[metric_name] += score
                row.append(score)
                pbar.set_description(f'{metric_name} of {img_name}: {score}')
                pbar.write(f'{metric_name} of {img_name}: {score}')
            else:
                assert os.path.isdir(args.input), 'input path must be a folder for FID.'
                score = iqa_model(args.input, args.ref)
                avg_scores[metric_name] += score
                row.append(score)
                pbar.set_description(f'{metric_name} of {img_name}: {score}')
                pbar.write(f'{metric_name} of {img_name}: {score}')

        if args.save_file:
            sfwriter.writerow(row)
        pbar.update(1)

    pbar.close()
    for metric_name in avg_scores:
        avg_scores[metric_name] /= test_img_num

    for metric_name, avg_score in avg_scores.items():
        msg = f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}'
        print(msg)
        if args.save_file:
            sf.write(f'{msg}\n')

    if args.save_file:
        sf.close()
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')


if __name__ == '__main__':
    main()
