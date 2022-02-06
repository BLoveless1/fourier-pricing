from pricer.core.request_handler import price_barrier_trades
from pricer.core.request_handler import price_alpha_quantile_trades
import json
import sys
import argparse


def main():
    # 0 - off, 1 - on
    Barrier = 0
    AlphaQuantile = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('barrier_requests')
    parser.add_argument('alpha_quantile_requests')
    args = parser.parse_args()

    if Barrier == 1:
        with open(args.barrier_requests) as requests:
            price_barrier_trades(
                json.load(requests),
                sys.stdout
            )

    if AlphaQuantile == 1:
        with open(args.alpha_quantile_requests) as requests:
            price_alpha_quantile_trades(
                json.load(requests),
                sys.stdout
            )

    sys.exit()


if __name__ == "__main__":
    main()
