import argparse
import json
import logging
import os

# keywords to be excluded from the final gathered results
EXCLUDE_LIST = ['eta', 'eta_hours', 'loss', 'lr']


def parse_args():
    """Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(
        description='Aggregate all results a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--result_dir', default='results',
                        help='Result directory')
    parser.add_argument('--output_dir', default='aggr_results',
                        help='Output directory')
    parser.add_argument('--dataset_name', default='zinc',
                        help='Dataset to collect results for')
    parser.add_argument('--output_fn', default='results.json',
                        help='Output filename')

    return parser.parse_args()


def iterate_over_settings(args):
    """Iterate over the result directories.

    Iterate over dataset(s) and configs and yield the
    corresponding result directory, which contains three sub directories:
    ``train``, ``val``, and ``test``. The target file is ``best.json``, which
    contains one line json about the aggregated evaluation of the dataset
    across all runs (different seeds).

    The result directory structure looks something like:
    <result_dir>/<dataset_name>/<config_name>/agg

    """
    # for dataset_name in [x for x in os.listdir(args.result_dir) if os.path.isdir(os.path.join(args.result_dir, x))]:
    dataset_name = args.dataset_name
    ds_dir = os.path.join(args.result_dir, dataset_name)
    for config_name in [x for x in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, x))]:
        agg_dir = os.path.join(ds_dir, config_name, 'agg')
        yield dataset_name, config_name, agg_dir


def add_result(results, scores, dataset_name, config_name, split):
    """Filter and add result to the result list.

    Given the dictionary ``scores`` of aggregated evaluations, exclude the
    standard deviation values and also the exclude keywords, then append the
    filtered results along with the information of the experiment to ``results``

    Args:
        results (list): list of final aggregated evaluations
        scores (dict): dictionary of aggregated evaluations for a specific
            dataset
        dataset_name (str): name of the dataset
        config_name (str): name of the config
        split (str): ``'train'``, ``'val'``, or ``'test'``

    """
    exclude_list = EXCLUDE_LIST + [x + '_std' for x in EXCLUDE_LIST]
    new_result = {}
    new_result['Dataset'] = dataset_name
    new_result['Config'] = config_name
    new_result['Split'] = split
    for kw, val in scores.items():
        if kw in exclude_list:  # or kw.endswith('_std') also exclude std values
            continue
        new_result[f'score-{kw}'] = val
    results.append(new_result)


def _print_elements(name, elements):
    print(f'Total number of {name}: {len(elements)}')
    for element in elements:
        print(f'    {element}')


def main():
    """Main function for result aggregation script.
    """
    args = parse_args()
    datasets = set()
    configs = set()

    results = []
    for dataset_name, config_name, agg_dir in iterate_over_settings(args):
        datasets.add(dataset_name)
        configs.add(config_name)
        logging.info(f'Loading results from: {agg_dir}')

        for split in ['train', 'val', 'test']:
            best_agg_fp = os.path.join(agg_dir, split, 'best.json')
            if not os.path.isfile(best_agg_fp):
                logging.warning(f'File does not exist: {best_agg_fp!r}')
                continue
            with open(best_agg_fp, 'r') as f:
                try:
                    scores = json.load(f)
                    add_result(results, scores, dataset_name, config_name, split)
                except Exception as e:
                    logging.warning(f'Issue with reading file: {best_agg_fp}')
                    raise e

    with open(f'{args.output_dir}/{args.output_fn}', 'w') as f:
        json.dump(results, f, indent=4)

    _print_elements('datasets', datasets)
    _print_elements('configs', configs)


if __name__ == "__main__":
    main()
