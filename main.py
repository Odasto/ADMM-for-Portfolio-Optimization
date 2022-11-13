#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time

import argparse

# symbols = data['symbols']


def main():
    args = parse_args()
    data = np.load(args.filename)
    Sigma = data['cov']
    mu = data['mean']

    start = time.time()
    res = admm(Sigma, mu, N=args.N, epsilon=args.epsilon, rho=args.rho)
    stop = time.time()

    print(f'Terminated in {(stop - start):.4f}sec\n')
    x_formated = str(res[0]).replace("\n", "\n    ")
    print(f'x = {x_formated}')
    print(f'Expected Risk = {res[0].T @ Sigma @ res[0]}')
    print(f'Expected Profit = {mu.T @ res[0]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='path to instance')
    parser.add_argument('-N', type=int, default=1000, help='number of iterations')
    parser.add_argument('-epsilon', type=float, default=0.0001, help='epsilon')
    parser.add_argument('-rho', type=float, default=1, help='rho')

    args = parser.parse_args()
    if args.N < 1:
        raise argparse.ArgumentTypeError('The number of iterations must be a positive integer.')
    if args.epsilon <= 0 or args.epsilon >= 1:
        raise argparse.ArgumentTypeError('The epsilon must be between 0 and 1.')
    if args.rho <= 0:
        raise argparse.ArgumentTypeError('The rho must be greater than 0.')

    return args


# ADMM algorithm
def admm(Sigma, mu, epsilon=0.0001, N=1000, rho=1):  # epsilon, N, rho as optional args with default values
    x = np.zeros((2, mu.shape[0]))
    y = np.zeros((2, mu.shape[0]))
    z = np.zeros((2, mu.shape[0]))
    s_norm = []
    r_norm = []
    value = []

    for i in range(0, N):
        k = i % 2
        k_pre = (i - 1) % 2

        A = 2 * Sigma + rho * np.identity(mu.shape[0])
        b = rho * (z[k_pre] - (1 / rho) * y[k_pre]) + mu
        x[k] = linalg.solve(A, b)  # A*x = b

        v = x[k] + (1 / rho) * y[k_pre]
        z[k] = v + ((1 - np.ones(mu.shape[0]).T @ v) / mu.shape[0]) * np.ones(mu.shape[0])

        y[k] = y[k_pre] + rho * (x[k] - z[k])

        s = rho * (z[k] - z[k_pre])
        s_norm.append(np.linalg.norm(s))
        r = x[k_pre] - z[k_pre]
        r_norm.append(np.linalg.norm(r))

        value.append(x[k].T @ Sigma @ x[k] - mu.T @ x[k])

        print(f'{i + 1}. value = {value[i]}, s_norm = {s_norm[i]}, r_norm = {r_norm[i]}\n')  # worsens runtime

        if s_norm[i] <= epsilon and r_norm[i] <= epsilon:
            return x[k], value, s_norm, r_norm, i + 1

    return x[(N - 1) % 2], value, s_norm, r_norm, N


if __name__ == "__main__":  #run main if program is executed stand alone
    main()
