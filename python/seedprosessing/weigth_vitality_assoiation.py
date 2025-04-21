import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

excel_file_path = 'sorted_output.xlsx'
data_path = 'entire_data.xlsx'

data0 = pd.read_excel(data_path, sheet_name='Sheet1')
data0 = np.array(data0, dtype=np.float64)
all_sheets = pd.read_excel(excel_file_path, sheet_name=None, header=None)
ori_data = np.concatenate((all_sheets['2'], all_sheets['3'], all_sheets['4'],
                           all_sheets['5'], all_sheets['6'], all_sheets['7'],
                           all_sheets['8'], all_sheets['9'], all_sheets['10'],
                           all_sheets['11']), axis=0)
data = np.concatenate((ori_data[:, 1:8], ori_data[:, 11].reshape(-1, 1)), axis=1)
data = np.array(data, dtype=np.float64)


data_vitality = data[:, 7]
weight = data[:, 0]
area = data0[:,8]
primeter = data0[:,9]




def analyze_association(weight_data, vitality_data):

    df = pd.DataFrame({
        'weight': weight_data,
        'vitality': vitality_data
    })

    # Basic statistics
    stats_summary = {
        'weight_stats': df['weight'].describe(),
        'vitality_distribution': df['vitality'].value_counts(normalize=True)
    }

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x='vitality', y='weight', data=df)
    plt.title('primeter Distribution by vitality Group')
    plt.xlabel('vitality')
    plt.ylabel('primeter')

    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='weight', hue='vitality', element='step', stat='density', common_norm=False)
    plt.title('primeter Distribution Overlay')
    plt.xlabel('primeter')

    plt.tight_layout()
    plt.savefig("weight_vitality_distribution",dpi=600)
    plt.show()

    # Statistical tests
    group0 = df[df['vitality'] == 0]['weight']
    group1 = df[df['vitality'] == 1]['weight']

    # Normality tests
    norm_test = {
        'group0': stats.shapiro(group0),
        'group1': stats.shapiro(group1)
    }

    # Group comparison
    if norm_test['group0'].pvalue > 0.05 and norm_test['group1'].pvalue > 0.05:
        test_result = stats.ttest_ind(group0, group1)
        test_used = 't-test'
    else:
        test_result = stats.mannwhitneyu(group0, group1)
        test_used = 'Mann-Whitney U'

    # Correlation analysis
    point_biserial = stats.pointbiserialr(df['vitality'], df['weight'])

    return {
        'summary_stats': stats_summary,
        'normality_tests': {
            'group0': {'W': norm_test['group0'].statistic, 'p': norm_test['group0'].pvalue},
            'group1': {'W': norm_test['group1'].statistic, 'p': norm_test['group1'].pvalue}
        },
        'group_comparison': {
            'test': test_used,
            'statistic': test_result.statistic,
            'p_value': test_result.pvalue
        },
        'correlation': {
            'point_biserial': point_biserial.correlation,
            'p_value': point_biserial.pvalue
        }
    }


# Example usage
if __name__ == "__main__":
    # Generate sample data

    # Run analysis
    results = analyze_association(primeter, data_vitality)

    # Print results
    print("=== Basic Statistics ===")
    print("Weight statistics:")
    print(results['summary_stats']['weight_stats'])
    print("\nBinary distribution:")
    print(results['summary_stats']['binary_distribution'])

    print("\n=== Normality Tests ===")
    print(
        f"Group 0: W={results['normality_tests']['group0']['W']:.3f}, p={results['normality_tests']['group0']['p']:.3f}")
    print(
        f"Group 1: W={results['normality_tests']['group1']['W']:.3f}, p={results['normality_tests']['group1']['p']:.3f}")

    print("\n=== Group Comparison ===")
    print(f"Test used: {results['group_comparison']['test']}")
    print(f"Statistic: {results['group_comparison']['statistic']:.3f}")
    print(f"P-value: {results['group_comparison']['p_value']:.3f}")

    print("\n=== Correlation Analysis ===")
    print(f"Point-biserial correlation: {results['correlation']['point_biserial']:.3f}")
    print(f"P-value: {results['correlation']['p_value']:.3f}")