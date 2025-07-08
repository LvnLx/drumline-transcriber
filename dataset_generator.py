from argparse import ArgumentParser
from random import randint

from tqdm import tqdm

import util.chunk_extractor as chunk_extractor
from exception.sampling_rate_mismatch_error import SamplingRateMismatchError
import util.file as file
from model.instrument import INSTRUMENTS


def main():
    parser = ArgumentParser(description="Generate dataset of short clips of marching percussion instrument hits")
    parser.add_argument("-s", "--singles", action="store_true", help="generate files for each instrument individually")
    parser.add_argument("-d", "--doubles", action="store_true", help="generate files for combinations of two instruments")
    parser.add_argument("-t", "--triples", action="store_true", help="generate files for combinations of three instruments")
    parser.add_argument("--divisor", type=int, default=1, metavar="", help="divides the number double or triple combinations by 2^divisor (default = 1)")
    args = parser.parse_args()

    if args.singles:
        for instrument in INSTRUMENTS:
            for path in tqdm(file.get_files("audio/{}".format(instrument.name))):
                chunk_extractor.extract_chunks(instrument, file.get_file_name(path))

    if args.doubles:
        files_created = 0
        singles = get_singles()
        for i in range(len(singles) - 1):
            for j in range(i + 1, len(singles)):
                for k in tqdm(range(randint(0, args.divisor - 1), len(singles[i]), args.divisor)):
                    for l in range(randint(0, args.divisor - 1), len(singles[j]), args.divisor):
                        if (singles[i][k][1] != singles[j][l][1]):
                            raise SamplingRateMismatchError(i, j, k, l, singles[i][k][1], singles[j][l][1])
                        y = (singles[i][k][0] + singles[j][l][0]) / 2
                        chunk_extractor.save_chunk("mixed", "{}_clip_{}_{}_clip_{}".format(INSTRUMENTS[i].name, k, INSTRUMENTS[j].name, l), y, singles[i][k][1], [INSTRUMENTS[i].label, INSTRUMENTS[j].label])
                        files_created += 1
        print("Combinations of two created: {}".format(files_created))

    if args.triples:
        files_created = 0
        singles = get_singles()
        for i in range(len(singles) - 2):
            for j in range(i + 1, len(singles) - 1):
                for k in range(i + 2, len(singles)):
                    for l in tqdm(range(randint(0, args.divisor - 1), len(singles[i]), args.divisor)):
                        for m in range(randint(0, args.divisor - 1), len(singles[j]), args.divisor):
                            for n in range(randint(0, args.divisor - 1), len(singles[k]), args.divisor):
                                if (singles[i][l][1] != singles[j][m][1]):
                                    raise SamplingRateMismatchError(i, j, l, m, singles[i][l][1], singles[j][m][1])
                                if (singles[i][l][1] != singles[k][n][1]):
                                    raise SamplingRateMismatchError(i, j, l, m, singles[i][l][1], singles[k][n][1])
                                y = (singles[i][l][0] + singles[j][m][0] + singles[k][n][0]) / 3
                                chunk_extractor.save_chunk("mixed", "{}_clip_{}_{}_clip_{}_{}_clip_{}".format(INSTRUMENTS[i].name, l, INSTRUMENTS[j].name, m, INSTRUMENTS[k], n), y, singles[i][l][1], [INSTRUMENTS[i].label, INSTRUMENTS[j].label, INSTRUMENTS[k].label])
                                files_created += 1
        print("Combintations of three created: {}".format(files_created))


def get_singles():
    audio_files = tuple([] for i in range(len(INSTRUMENTS)))
    for i in range(len(audio_files)):
        for path in file.get_files("audio/extracted/{}/audio".format(INSTRUMENTS[i].name)):
            audio_files[i].append(file.load_audio(path))

    return audio_files


if __name__ == "__main__":
    main()
