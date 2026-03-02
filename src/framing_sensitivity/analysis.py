from statsmodels.stats.contingency_tables import mcnemar


def exact_mcnemar_test(table: list[list[int]]) -> tuple[float, float]:
    result = mcnemar(table, exact=True, correction=False)
    return float(result.statistic), float(result.pvalue)
