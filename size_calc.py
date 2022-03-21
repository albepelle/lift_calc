import pandas as pd
import numpy as np
import statsmodels.stats as st
import seaborn as ns
import matplotlib.pyplot as plt


def max_even(data):
    L = []
    for i in reversed(range(len(data), 0, -1)):
        if (i % 2) == 0:
            L.append(i)
    return max (L)


def s_inc(split_group):

    split_group=split_group.head(max_even(split_group))
    print('your sample size has been reduced to:'+str(len(split_group))+' in order to ensure a 50% split')
    ggr_gbp = split_group.values

    # define bootstrapping function
    def bootstrap_replicate_1d(data, func):
        """Generate bootstrap replicate of 1D data."""

        bs_sample = np.random.choice(data, len(data))
        return func(bs_sample)

    # create empty dict for the results
    test_results = {'increase': [], 'mean control': [], 'std_control': [], 'std_test': [], 'mean test': [],
                    'mean delta': [],
                    'delta LB': [], 'delta UB': [], 'P(T>C)': []}

    # define how many itterations in bootstrapping
    samples = 10000

    bs_replicates_test = np.empty(samples)
    bs_replicates_control = np.empty(samples)

    # populate the array with mean values from the test group

    import tqdm as tn
    for n in tn.tqdm_notebook(range(5, 60, 5)):
        for i in tn.tqdm_notebook(range(samples), leave=bool(n == 55)):
            np.random.shuffle(ggr_gbp)

            a = ggr_gbp[:int(len(ggr_gbp) / 2)]
            b = ggr_gbp[int(len(ggr_gbp) / 2) + 1:]

            a = a * (1 + (n / 100))

            # print(np.random.choice(a))

            bs_replicates_test[i] = bootstrap_replicate_1d(np.array(a).flatten(),
                                                           np.mean)

            bs_replicates_control[i] = bootstrap_replicate_1d(np.array(b).flatten(),
                                                              np.mean)

            # calculate the mean of the test group
            mean_test = round(np.mean(bs_replicates_test), 2)
            std_test = round(bootstrap_replicate_1d(np.array(a).flatten(), np.std), 2)

            # calculate the mean of the control group
            mean_control = round(np.mean(bs_replicates_control), 2)
            std_control = round(bootstrap_replicate_1d(np.array(b).flatten(), np.std), 2)

            # calculate the difference in the means (test minus control)
            difference_in_mean = bs_replicates_test - bs_replicates_control
            mean_diff_in_mean = round(np.mean(difference_in_mean), 2)

            # calculate the 95% confidence intervals in the difference of the means
            cfl_diff_in_mean = round(np.percentile(difference_in_mean, 2.5), 2)
            cfu_diff_in_mean = round(np.percentile(difference_in_mean, 97.5), 2)

            # calculate P(T>C)
            prob_test_higher_control = round(np.sum(bs_replicates_test > bs_replicates_control) / samples, 2)

        # append the results to the dictionary
        # test_results['metric'].append(metric)
        test_results['increase'].append(n)
        test_results['mean control'].append(mean_control)
        test_results['std_control'].append(std_control)
        test_results['mean test'].append(mean_test)
        test_results['std_test'].append(std_test)
        test_results['mean delta'].append(mean_diff_in_mean)
        test_results['delta LB'].append(cfl_diff_in_mean)
        test_results['delta UB'].append(cfu_diff_in_mean)
        test_results['P(T>C)'].append(prob_test_higher_control)

    # create the results dataframe from the results dictionary
    df_test_results = pd.DataFrame.from_dict(test_results)

    # plotting

    fig = df_test_results[['increase', '''P(T>C)''']].set_index('increase').plot()
    plt.ylabel('Probability Test > Control')
    plt.xlabel('Lift %')


    IND = min(range(len(df_test_results['''P(T>C)'''])), key=lambda i: abs(df_test_results['''P(T>C)'''][i] - 0.9))
    plt.axhline(0.9, color='darkblue', linestyle="--")

    plt.axvline(df_test_results['increase'][IND], color='darkblue', linestyle="--")
    plt.title('probability test vs control with sample size of: ' + str(int(len(ggr_gbp) / 2)))

    return df_test_results


def s_size(split_group, increase):
    split_group = split_group.head(max_even(split_group))

    print('your sample size has been reduced to:'+str(len(split_group))+' in order to ensure a 50% split')
    ggr_gbp = split_group.values

    # define bootstrapping function
    def bootstrap_replicate_1d(data, func):
        """Generate bootstrap replicate of 1D data."""

        bs_sample = np.random.choice(data, len(data))
        return func(bs_sample)

    # create empty dict for the results
    test_results = {'sample size': [], 'mean control': [], 'std_control': [], 'std_test': [], 'mean test': [],
                    'mean delta': [],
                    'delta LB': [], 'delta UB': [], 'P(T>C)': []}

    # define how many itterations in bootstrapping
    samples = 10000

    bs_replicates_test = np.empty(samples)
    bs_replicates_control = np.empty(samples)

    # populate the array with mean values from the test group

    def gtd(d):
        # 1
        num = int(d)
        largest_divisor = 0

        # 2
        for i in range(2, num):
            # 3
            if num % i == 0:
                # 4
                largest_divisor = i

        # 5
        print("Largest divisor of {} is {}".format(num, largest_divisor))

        print('\n')

        print('would you like to proceed with this sample size increase? Y/N')

        answ = input()

        if answ.upper() == 'Y':

            ret = largest_divisor


        else:
            print('please input an alternative sample increase..')
            ret = int(input())

        return (ret)

    sample_inc = gtd(len(ggr_gbp) / 2)

    import tqdm as tn

    for n in tn.tqdm_notebook(range(sample_inc, int(len(ggr_gbp) / 2), sample_inc)):
        for i in tn.tqdm_notebook(range(samples), leave=bool(n == len(ggr_gbp) / 2)):
            np.random.shuffle(ggr_gbp)

            a = ggr_gbp[:n]
            b = ggr_gbp[n + 1:n * 2]

            a = a * (1 + increase)

            # print(np.random.choice(a))

            bs_replicates_test[i] = bootstrap_replicate_1d(np.array(a).flatten(),
                                                           np.mean)

            bs_replicates_control[i] = bootstrap_replicate_1d(np.array(b).flatten(),
                                                              np.mean)

            # calculate the mean of the test group
            mean_test = round(np.mean(bs_replicates_test), 2)
            std_test = round(bootstrap_replicate_1d(np.array(a).flatten(), np.std), 2)

            # calculate the mean of the control group
            mean_control = round(np.mean(bs_replicates_control), 2)
            std_control = round(bootstrap_replicate_1d(np.array(b).flatten(), np.std), 2)

            # calculate the difference in the means (test minus control)
            difference_in_mean = bs_replicates_test - bs_replicates_control
            mean_diff_in_mean = round(np.mean(difference_in_mean), 2)

            # calculate the 95% confidence intervals in the difference of the means
            cfl_diff_in_mean = round(np.percentile(difference_in_mean, 2.5), 2)
            cfu_diff_in_mean = round(np.percentile(difference_in_mean, 97.5), 2)

            # calculate P(T>C)
            prob_test_higher_control = round(np.sum(bs_replicates_test > bs_replicates_control) / samples, 2)

        # append the results to the dictionary
        # test_results['metric'].append(metric)
        test_results['sample size'].append(n)
        test_results['mean control'].append(mean_control)
        test_results['std_control'].append(std_control)
        test_results['mean test'].append(mean_test)
        test_results['std_test'].append(std_test)
        test_results['mean delta'].append(mean_diff_in_mean)
        test_results['delta LB'].append(cfl_diff_in_mean)
        test_results['delta UB'].append(cfu_diff_in_mean)
        test_results['P(T>C)'].append(prob_test_higher_control)

    # create the results dataframe from the results dictionary
    df_test_results = pd.DataFrame.from_dict(test_results)

    # plotting

    fig = df_test_results[['sample size', '''P(T>C)''']].set_index('sample size').plot()
    plt.ylabel('Probability Test > Control')
    plt.xlabel('sample size')
    IND = min(range(len(df_test_results['''P(T>C)'''])), key=lambda i: abs(df_test_results['''P(T>C)'''][i] - 0.9))
    plt.axhline(0.9, color='darkblue', linestyle="--")

    plt.axvline(df_test_results['sample size'][IND], color='darkblue', linestyle="--")
    plt.title('probability test vs control with a lift%: ' + str(increase * 100) + '%')

    return df_test_results



