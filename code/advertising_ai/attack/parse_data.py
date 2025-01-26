import sys
import os
import webdataset as wds
import time
import random, string
import pickle
import argparse
import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'download_images')))

def pad_zeros(idx):
    idx_str = str(idx)
    if len(idx_str) < 5:
        idx_str = "0" * (5 - len(idx_str)) + idx_str
    return idx_str


def main():
    get_code = lambda: ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    start_id = args.idx
    # end_id = start_id + 1000
    end_id = args.idx_end

    start_id = pad_zeros(start_id)
    end_id = pad_zeros(end_id)

    url = "{}/{{{}..{}}}.tar".format(
        args.input_dir, start_id, end_id)
    print(url)
    dataset = wds.WebDataset(url).to_tuple("txt", "jpg")

    total = 0
    found = 0
    os.makedirs(f"data/{args.concept}", exist_ok=True)
    for idx, i in enumerate(dataset):
        # if random.uniform(0, 1) > 0.001:
        #     continue
        total += 1
        text = i[0].decode()
        cur_text = text.lower()
        if total % 20000 == 0:
            print("Total: {} | Found: {}".format(total, found))

        if f" {args.concept} " not in cur_text:
            continue

        img = i[1]
        found += 1

        res = {"text": text, "img": img}
        code = get_code()

        pickle.dump(res, open("data/{}/{}.p".format(args.concept, code), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int,
                        default=14,
                        required=True,
                        )
    parser.add_argument('--idx_end', type=int,
                        default=199,
                        required=True,
                        )
    parser.add_argument('--concept', type=str, default='dog')
    parser.add_argument('--input_dir', type=str, default='../download_images/archive/laion400m-subset')
    return parser.parse_args(argv)


if __name__ == '__main__':
    t = time.time()
    args = parse_arguments(sys.argv[1:])
    main()
    t_total = time.time() - t
    print("Time: {}".format(t_total))
